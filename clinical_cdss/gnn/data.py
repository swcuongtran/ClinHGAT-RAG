from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from clinical_cdss.core.database import Neo4jConnection


PATIENT_FEATURES = ["age", "gender", "weight", "is_child", "admission_day"]
EVIDENCE_FEATURES = [
    "plt_nadir",
    "plt_below10",
    "hct_peak",
    "hct_change_pct",
    "wbc_trend",
    "hflc_peak",
    "ast_value",
    "ast_available",
    "alt_value",
    "creatinine_value",
    "critical_day",
]


@dataclass
class ClinicalGraphData:
    x_dict: Dict[str, torch.Tensor]
    labels: torch.Tensor
    patient_ids: List[str]
    node_names: List[str]
    node_types: List[str]
    node_type_indices: Dict[str, torch.Tensor]
    patient_node_indices: torch.Tensor
    evidence_node_indices: torch.Tensor
    evidence_ids: List[str]
    hyperedge_names: List[str]
    incidence: torch.Tensor
    patient_to_evidence: Dict[int, int]


def _float_value(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return float(value)


def _embedding(value, dim: int = 768):
    if value is None:
        return [0.0] * dim
    if len(value) >= dim:
        return [float(v) for v in value[:dim]]
    return [float(v) for v in value] + [0.0] * (dim - len(value))


def _tensor2d(rows, cols: int):
    if not rows:
        return torch.empty((0, cols), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def load_graph_data() -> ClinicalGraphData:
    db = Neo4jConnection()
    try:
        patients = db.execute_query("""
            MATCH (p:Patient)
            WHERE p.diagnosis_label IS NOT NULL
            RETURN p.id AS id, p.diagnosis_label AS label,
                   p.age AS age, p.gender AS gender, p.weight AS weight,
                   p.is_child AS is_child, p.admission_day AS admission_day
            ORDER BY p.id
        """)
        evidences = db.execute_query("""
            MATCH (p:Patient)-[:HAS_EVIDENCE]->(ec:EvidenceCase)
            WHERE p.diagnosis_label IS NOT NULL
            RETURN p.id AS patient_id, ec.id AS id,
                   ec.plt_nadir AS plt_nadir,
                   ec.plt_below10 AS plt_below10,
                   ec.hct_peak AS hct_peak,
                   ec.hct_change_pct AS hct_change_pct,
                   ec.wbc_trend AS wbc_trend,
                   ec.hflc_peak AS hflc_peak,
                   ec.ast_value AS ast_value,
                   ec.ast_available AS ast_available,
                   ec.alt_value AS alt_value,
                   ec.creatinine_value AS creatinine_value,
                   ec.critical_day AS critical_day
            ORDER BY p.id
        """)
        symptoms = db.execute_query("""
            MATCH (s:Symptom)
            RETURN s.name AS name, s.embedding AS embedding
            ORDER BY s.name
        """)
        concepts = db.execute_query("""
            MATCH (c:Concept)
            RETURN c.name AS name
            ORDER BY c.name
        """)
        rules = db.execute_query("""
            MATCH (r:Diagnostic_Rule)
            RETURN r.name AS name, r.embedding AS embedding
            ORDER BY r.name
        """)
        ec_symptoms = db.execute_query("""
            MATCH (ec:EvidenceCase)-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN ec.id AS ec_id, s.name AS name
        """)
        ec_concepts = db.execute_query("""
            MATCH (ec:EvidenceCase)-[:HAS_CONCEPT]->(c:Concept)
            RETURN ec.id AS ec_id, c.name AS name
        """)
        ec_rules = db.execute_query("""
            MATCH (ec:EvidenceCase)-[:MATCHES]->(r:Diagnostic_Rule)
            RETURN ec.id AS ec_id, r.name AS name
        """)
        rule_symptoms = db.execute_query("""
            MATCH (s:Symptom)-[:PART_OF_RULE]->(r:Diagnostic_Rule)
            RETURN r.name AS rule_name, s.name AS name
        """)
        rule_concepts = db.execute_query("""
            MATCH (c:Concept)-[:LINKED_TO_RULE]->(r:Diagnostic_Rule)
            RETURN r.name AS rule_name, c.name AS name
        """)
    finally:
        db.close()

    node_names: List[str] = []
    node_types: List[str] = []
    global_index: Dict[Tuple[str, str], int] = {}

    def add_node(node_type: str, name: str):
        global_index[(node_type, name)] = len(node_names)
        node_names.append(name)
        node_types.append(node_type)

    for row in patients:
        add_node("patient", row["id"])
    for row in evidences:
        add_node("evidence", row["id"])
    for row in symptoms:
        add_node("symptom", row["name"])
    for row in concepts:
        add_node("concept", row["name"])
    for row in rules:
        add_node("rule", row["name"])

    patient_ids = [row["id"] for row in patients]
    evidence_ids = [row["id"] for row in evidences]
    patient_features = [
        [_float_value(row[field]) for field in PATIENT_FEATURES]
        for row in patients
    ]
    evidence_features = [
        [_float_value(row[field]) for field in EVIDENCE_FEATURES]
        for row in evidences
    ]
    symptom_features = [_embedding(row["embedding"]) for row in symptoms]
    # Represent each concept as a One-Hot vector (learnable distinct embeddings).
    concept_features = np.eye(len(concepts)).tolist()
    rule_features = [_embedding(row["embedding"]) for row in rules]

    x_dict = {
        "patient": _tensor2d(patient_features, len(PATIENT_FEATURES)),
        "evidence": _tensor2d(evidence_features, len(EVIDENCE_FEATURES)),
        "symptom": _tensor2d(symptom_features, 768),
        "concept": _tensor2d(concept_features, len(concepts)),
        "rule": _tensor2d(rule_features, 768),
    }
    labels = torch.tensor([int(row["label"]) - 1 for row in patients], dtype=torch.long)

    hyperedges: List[Tuple[str, List[int]]] = []
    patient_to_evidence: Dict[int, int] = {}
    patient_by_id = {row["id"]: i for i, row in enumerate(patients)}
    evidence_patient = {row["id"]: row["patient_id"] for row in evidences}

    ec_to_symptoms: Dict[str, List[str]] = {}
    ec_to_concepts: Dict[str, List[str]] = {}
    ec_to_rules: Dict[str, List[str]] = {}
    rule_to_symptoms: Dict[str, List[str]] = {}
    rule_to_concepts: Dict[str, List[str]] = {}

    for row in ec_symptoms:
        ec_to_symptoms.setdefault(row["ec_id"], []).append(row["name"])
    for row in ec_concepts:
        ec_to_concepts.setdefault(row["ec_id"], []).append(row["name"])
    for row in ec_rules:
        ec_to_rules.setdefault(row["ec_id"], []).append(row["name"])
    for row in rule_symptoms:
        rule_to_symptoms.setdefault(row["rule_name"], []).append(row["name"])
    for row in rule_concepts:
        rule_to_concepts.setdefault(row["rule_name"], []).append(row["name"])

    for ec_id in evidence_ids:
        nodes = [global_index[("evidence", ec_id)]]
        patient_id = evidence_patient.get(ec_id)
        if patient_id in patient_by_id:
            nodes.append(global_index[("patient", patient_id)])
            patient_to_evidence[patient_by_id[patient_id]] = global_index[("evidence", ec_id)]
        for name in ec_to_symptoms.get(ec_id, []):
            idx = global_index.get(("symptom", name))
            if idx is not None:
                nodes.append(idx)
        for name in ec_to_concepts.get(ec_id, []):
            idx = global_index.get(("concept", name))
            if idx is not None:
                nodes.append(idx)
        for name in ec_to_rules.get(ec_id, []):
            idx = global_index.get(("rule", name))
            if idx is not None:
                nodes.append(idx)
        hyperedges.append((f"evidence:{ec_id}", sorted(set(nodes))))

    for row in rules:
        rule_name = row["name"]
        nodes = [global_index[("rule", rule_name)]]
        for name in rule_to_symptoms.get(rule_name, []):
            idx = global_index.get(("symptom", name))
            if idx is not None:
                nodes.append(idx)
        for name in rule_to_concepts.get(rule_name, []):
            idx = global_index.get(("concept", name))
            if idx is not None:
                nodes.append(idx)
        hyperedges.append((f"rule:{rule_name}", sorted(set(nodes))))

    incidence = torch.zeros((len(node_names), len(hyperedges)), dtype=torch.float32)
    for edge_idx, (_, nodes) in enumerate(hyperedges):
        for node_idx in nodes:
            incidence[node_idx, edge_idx] = 1.0

    node_type_indices = {
        node_type: torch.tensor(
            [idx for idx, value in enumerate(node_types) if value == node_type],
            dtype=torch.long,
        )
        for node_type in ["patient", "evidence", "symptom", "concept", "rule"]
    }
    patient_node_indices = node_type_indices["patient"]
    evidence_node_indices = node_type_indices["evidence"]

    return ClinicalGraphData(
        x_dict=x_dict,
        labels=labels,
        patient_ids=patient_ids,
        node_names=node_names,
        node_types=node_types,
        node_type_indices=node_type_indices,
        patient_node_indices=patient_node_indices,
        evidence_node_indices=evidence_node_indices,
        evidence_ids=evidence_ids,
        hyperedge_names=[name for name, _ in hyperedges],
        incidence=incidence,
        patient_to_evidence=patient_to_evidence,
    )
