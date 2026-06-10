from typing import Dict, List, Optional, Tuple

import torch

from clinical_cdss.core.database import Neo4jConnection


def top_evidence_case(data, model_output, patient_idx: int) -> Tuple[Optional[str], float]:
    """Return the EvidenceCase hyperedge most connected to a patient by attention."""
    beta = model_output["attention_beta"].detach().cpu()
    patient_node_idx = int(data.patient_node_indices[patient_idx])
    patient_weights = beta[patient_node_idx]

    evidence_candidates = [
        (edge_idx, name)
        for edge_idx, name in enumerate(data.hyperedge_names)
        if name.startswith("evidence:")
    ]
    if not evidence_candidates:
        return None, 0.0

    best_edge_idx, best_name = max(
        evidence_candidates,
        key=lambda item: float(patient_weights[item[0]]),
    )
    return best_name.replace("evidence:", "", 1), float(patient_weights[best_edge_idx])


def top_attention_nodes(
    data,
    model_output,
    evidence_id: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    edge_name = f"evidence:{evidence_id}"
    if edge_name not in data.hyperedge_names:
        return []
    edge_idx = data.hyperedge_names.index(edge_name)
    alpha = model_output["attention_alpha"].detach().cpu()[:, edge_idx]
    member_mask = data.incidence[:, edge_idx] > 0
    weights = alpha.masked_fill(~member_mask, -1.0)
    values, indices = torch.topk(weights, k=min(k, int(member_mask.sum())))
    return [
        (data.node_names[int(idx)], float(value))
        for idx, value in zip(indices, values)
        if float(value) >= 0.0
    ]


def retrieve_subgraph(patient_id: str, evidence_id: Optional[str] = None) -> Dict:
    db = Neo4jConnection()
    try:
        if evidence_id is None:
            rows = db.execute_query("""
                MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)
                RETURN ec.id AS id
                LIMIT 1
            """, {"pid": patient_id})
            evidence_id = rows[0]["id"] if rows else None
        if evidence_id is None:
            return {"error": f"No EvidenceCase found for patient {patient_id}"}

        rows = db.execute_query("""
            MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase {id: $ec_id})
            OPTIONAL MATCH (ec)-[:HAS_SYMPTOM]->(s:Symptom)
            OPTIONAL MATCH (ec)-[:HAS_CONCEPT]->(c:Concept)
            OPTIONAL MATCH (ec)-[m:MATCHES]->(rule:Diagnostic_Rule)
            OPTIONAL MATCH (rule)-[:DETERMINES]->(sev:Severity)
            WITH ec, m, rule, sev,
                 collect(DISTINCT s.name) AS symptoms,
                 collect(DISTINCT c.name) AS concepts
            RETURN ec.id AS evidence_id,
                   ec.plt_nadir AS plt_nadir,
                   ec.hct_peak AS hct_peak,
                   ec.hflc_peak AS hflc_peak,
                   ec.ast_value AS ast_value,
                   ec.critical_day AS critical_day,
                   symptoms, concepts,
                   rule.name AS matched_rule,
                   sev.name AS severity,
                   m.coverage_score AS coverage_score
            ORDER BY coverage_score DESC,
                     CASE
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốc' AND (coalesce(toLower(sev.name), '') CONTAINS 'nặng' OR coalesce(toLower(sev.name), '') CONTAINS 'kéo dài' OR coalesce(toLower(sev.name), '') CONTAINS 'thất bại') THEN 5
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốc' THEN 4
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'cảnh báo' OR coalesce(toLower(sev.name), '') CONTAINS 'chuyển độ' THEN 3
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốt xuất huyết' OR coalesce(toLower(sev.name), '') CONTAINS 'sốt dengue nặng' THEN 2
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốt' OR coalesce(toLower(sev.name), '') CONTAINS 'chăm sóc' OR coalesce(toLower(sev.name), '') CONTAINS 'lo lắng' THEN 1
                          WHEN coalesce(toLower(sev.name), '') CONTAINS 'chưa phân loại' THEN 0
                          ELSE 1 END DESC,
                     rule.name ASC
            LIMIT 1
        """, {"pid": patient_id, "ec_id": evidence_id})
    finally:
        db.close()

    if not rows:
        return {"error": f"No XAI subgraph found for patient {patient_id}"}

    row = rows[0]
    evidence_summary = (
        f"PLT={row['plt_nadir']}, HCT={row['hct_peak']}, "
        f"HFLC={row['hflc_peak']}, AST={row['ast_value']}, "
        f"critical_day={row['critical_day']}"
    )
    return {
        "patient_id": patient_id,
        "evidence_id": row["evidence_id"],
        "evidence_summary": evidence_summary,
        "symptoms": row["symptoms"] or [],
        "concepts": row["concepts"] or [],
        "matched_rule": row["matched_rule"],
        "severity": row["severity"],
        "coverage_score": row["coverage_score"] or 0.0,
    }
