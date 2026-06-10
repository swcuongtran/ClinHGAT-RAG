# Ke hoach Trien khai - Clinical CDSS 3 tang

## Kien truc tong quan

```text
Benh nhan moi (CSV / Form)
        |
        v
Feature Engineering + EvidenceCase node
        |
        v
Tang 1 - HGAT chan doan
  EvidenceCase = hyperedge tuong minh
  Output: diagnosis, confidence, attention
        |
        v
Tang 2 - Graph Traversal XAI
  Patient -> EvidenceCase -> Symptom / Concept -> Diagnostic_Rule
  Diagnostic_Rule -> Severity / Treatment / Contraindication
        |
        v
Tang 3 - Guideline-grounded RAG
  Retrieve guideline chunk tu PDF BYT
  LLM viet bao cao co can cu
        |
        v
Bao cao lam sang + XAI path + guideline evidence
```

## Trang thai hien tai cua code

### Da co

| Hang muc | File | Ghi chu |
|---|---|---|
| EvidenceCase node | `etl_patients.py` | Da tao `EvidenceCase` cho moi benh nhan |
| Patient labels | `etl_patients.py` | Da co `diagnosis_label`, `binary_label` |
| Patient aggregate features | `etl_patients.py` | Da co `plt_nadir`, `hct_peak`, `hct_change_pct`, `hflc_peak`, `wbc_trend`, `ast_value`, `ast_available` |
| EvidenceCase -> Symptom | `etl_patients.py` | Da co `HAS_SYMPTOM` |
| EvidenceCase -> Concept | `etl_patients.py` | Da co `HAS_CONCEPT` |
| Concept -> Rule | `loader_guidelines.py` | Da co `LINKED_TO_RULE` |
| Similar patients | `etl_patients.py` | Da co `SIMILAR_TO` dua tren shared symptoms |
| Coverage fallback | `rag_engine.py` | Da co weighted coverage score Symptom/LabTest |

### Chua co

| Hang muc | File du kien | Ghi chu |
|---|---|---|
| Neo4j constraints/indexes | `database.py` hoac script moi | Can tao unique constraint cho cac node chinh |
| EvidenceCase -> Diagnostic_Rule | `etl_patients.py` hoac script moi | Can them `MATCHES` edge voi coverage score |
| GuidelineChunk nodes | `loader_guidelines.py` | Can index text chunk goc tu PDF |
| Vector index guideline chunks | `database.py` | Can `guideline_chunk_index` |
| Tensor builder | `gnn_data.py` | Neo4j -> tensors + incidence matrix H |
| HGAT model | `gnn_model.py` | ClinicalHGAT |
| Training loop | `gnn_train.py` | Train/evaluate voi cross-validation |
| Attention explanation | `gnn_explain.py` | attention -> top EvidenceCase -> subgraph |
| Confidence routing | `gnn_predict.py` | HGAT neu du tin cay, fallback coverage neu thap |
| HGAT/RAG integration | `rag_engine.py` | Nhan subgraph dict + guideline chunk |
| UI XAI | `app.py` | Attention heatmap + subgraph panel |

## Node EvidenceCase

`EvidenceCase` la hyperedge tuong minh, gom bang chung lam sang va can lam sang cua mot benh nhan.

```cypher
(Patient)-[:HAS_EVIDENCE]->(EvidenceCase)
(EvidenceCase)-[:HAS_SYMPTOM]->(Symptom)
(EvidenceCase)-[:HAS_CONCEPT]->(Concept)
(EvidenceCase)-[:MATCHES {coverage_score, sym_match, sym_total, concept_match, concept_total}]->(Diagnostic_Rule)
```

Vi du properties:

```python
{
    "plt_nadir": 18,
    "plt_below10": False,
    "hct_peak": 47.4,
    "hct_change_pct": 18.5,
    "hflc_peak": 0.65,
    "wbc_trend": -0.42,
    "ast_value": 719,
    "ast_available": 1,
    "critical_day": 5
}
```

## Hypergraph cho HGAT

### Node types

| Node | Feature |
|---|---|
| `Patient` | `[age, gender, weight, is_child, admission_day]` |
| `EvidenceCase` | `[plt_nadir, plt_below10, hct_peak, hct_change_pct, wbc_trend, hflc_peak, ast_value, ast_available]` |
| `Symptom` | SBERT embedding 768-dim |
| `Concept` | one-hot hoac learned embedding |
| `Diagnostic_Rule` | SBERT embedding cua `phase + severity + rule_name` |

### Edge types

| Edge | Nguon | Trang thai |
|---|---|---|
| `Patient -[:HAS_EVIDENCE]-> EvidenceCase` | ETL | Da co |
| `EvidenceCase -[:HAS_SYMPTOM]-> Symptom` | ETL | Da co |
| `EvidenceCase -[:HAS_CONCEPT]-> Concept` | ETL | Da co |
| `EvidenceCase -[:MATCHES]-> Diagnostic_Rule` | Computed | Can them |
| `Symptom -[:PART_OF_RULE]-> Diagnostic_Rule` | Guidelines | Da co |
| `Concept -[:LINKED_TO_RULE]-> Diagnostic_Rule` | Guidelines | Da co |
| `Diagnostic_Rule -[:DETERMINES]-> Severity` | Guidelines | Da co |
| `Diagnostic_Rule -[:RECOMMENDS]-> Treatment` | Guidelines | Da co |
| `Diagnostic_Rule -[:AVOIDS]-> Contraindication` | Guidelines | Da co |
| `Patient -[:SIMILAR_TO]-> Patient` | Computed | Da co |

## Tang 1 - HGAT

### Model du kien

```python
class ClinicalHGAT(nn.Module):
    def __init__(self, in_dim_dict, hidden=64, num_classes=2, heads=4):
        # Node encoders:
        #   Patient, EvidenceCase, Symptom, Concept, Diagnostic_Rule -> hidden dim
        # HGAT Layer 1: hidden=64, heads=4, dropout=0.3
        # HGAT Layer 2: hidden=32, heads=2, dropout=0.2
        # Classifier: Linear(32, num_classes), chi tinh loss tren Patient nodes
        ...
```

### Input/Output

```text
Input:
  X_patient       [n_patient, 5]
  X_evidence      [n_evidence, 8]
  X_symptom       [n_symptom, 768]
  X_concept       [n_concept, d]
  X_rule          [n_rule, 768]
  H_incidence     sparse [n_nodes, n_hyperedges]

Output:
  logits          [n_patient, 2]
  confidence      [n_patient]
  attention_alpha node -> hyperedge
  attention_beta  hyperedge -> node
```

### Luu y ve attention

Chi doc `attention_beta[:, patient_idx]` neu `Patient` that su nam trong incidence matrix cua hyperedge. Neu khong dua `Patient` vao hyperedge, can doc attention tu `EvidenceCase` cua benh nhan:

```python
patient_ec = patient_to_evidence_case[patient_idx]
top_hyperedges = attention_from_node(patient_ec)
```

## Tang 2 - Graph Traversal XAI

Dung top EvidenceCase tu HGAT de truy vet subgraph:

```cypher
MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)
WHERE ec.id = $top_ec_id
OPTIONAL MATCH (ec)-[:HAS_SYMPTOM]->(s:Symptom)
OPTIONAL MATCH (ec)-[:HAS_CONCEPT]->(c:Concept)
OPTIONAL MATCH (ec)-[m:MATCHES]->(rule:Diagnostic_Rule)
OPTIONAL MATCH (rule)-[:DETERMINES]->(sev:Severity)
OPTIONAL MATCH (rule)-[:RECOMMENDS]->(treat:Treatment)
OPTIONAL MATCH (rule)-[:AVOIDS]->(contra:Contraindication)
RETURN ec,
       collect(DISTINCT s.name) AS symptoms,
       collect(DISTINCT c.name) AS concepts,
       rule.name AS rule_name,
       sev.name AS severity,
       m.coverage_score AS coverage_score,
       collect(DISTINCT treat.action) AS treatments,
       collect(DISTINCT contra.action) AS contraindications
ORDER BY coverage_score DESC
LIMIT 1
```

Subgraph output:

```python
{
    "evidence_summary": "PLT=18, HCT=47.4%, Non oi, Xuat huyet",
    "symptoms": ["Non oi", "Xuat huyet"],
    "concepts": ["Co dac mau", "Giam tieu cau"],
    "matched_rule": "Soc Dengue (Giai doan nguy hiem)",
    "severity": "Soc Dengue",
    "coverage_score": 0.87,
    "treatments": ["Truyen Ringer Lactate ..."],
    "contraindications": ["Khong dung Aspirin"],
    "attention_nodes": [("Co dac mau", 0.61), ("Non oi", 0.31)]
}
```

## Tang 3 - Guideline-Grounded RAG

Can them node:

```cypher
(GuidelineChunk {
  id,
  text,
  source,
  page_start,
  page_end,
  embedding
})
```

Vector index:

```cypher
CREATE VECTOR INDEX guideline_chunk_index IF NOT EXISTS
FOR (g:GuidelineChunk)
ON (g.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
```

Retrieve:

```python
def retrieve_guideline_chunk(rule_name: str, concepts: list[str]) -> str:
    query = f"{rule_name} {' '.join(concepts)}"
    vector = embedder.encode(query).tolist()
    results = db.execute_query("""
        CALL db.index.vector.queryNodes('guideline_chunk_index', 3, $vector)
        YIELD node AS chunk, score
        WHERE score > 0.75
        RETURN chunk.text AS text, chunk.source AS source, score
        ORDER BY score DESC
    """, {"vector": vector})
    return "\n".join(r["text"] for r in results)
```

## Confidence-Gated Routing

```python
THRESHOLD = 0.75

def diagnose(patient_id: str):
    diagnosis, confidence, attention = hgat.predict(patient_id)

    if confidence >= THRESHOLD:
        top_ec = get_top_evidence_case(attention, patient_id)
        subgraph = graph_traversal(patient_id, top_ec)
        method = f"HGAT (conf={confidence:.0%})"
    else:
        subgraph = coverage_fallback(patient_id)
        method = f"Coverage Score fallback (conf={confidence:.0%})"

    guideline = retrieve_guideline_chunk(
        subgraph.get("matched_rule") or subgraph.get("rule_name", ""),
        subgraph.get("concepts", []),
    )
    report = generate_report(subgraph, guideline, method)
    similar_cases = find_similar_patients(patient_id)

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "method": method,
        "subgraph": subgraph,
        "report": report,
        "similar_cases": similar_cases,
    }
```

## Cac thay doi file du kien

```text
benh nhiet doi/
├── app.py                         <- Wrapper de chay Streamlit nhu cu
├── main.py                        <- CLI initialize/chat wrapper
├── requirements.txt
├── KE_HOACH_TRIEN_KHAI_CDSS.md
├── clinical_cdss/
│   ├── core/
│   │   ├── config.py              <- Env/config
│   │   └── database.py            <- Neo4j connection + schema/indexes
│   ├── etl/
│   │   ├── patients.py            <- Patient/EvidenceCase ETL + MATCHES
│   │   ├── guidelines.py          <- Rule loader + GuidelineChunk indexing
│   │   └── extractor_guidelines.py
│   ├── rag/
│   │   └── engine.py              <- Graph/RAG report engine
│   ├── gnn/
│   │   ├── data.py                <- Neo4j -> tensors + incidence matrix H
│   │   ├── model.py               <- ClinicalHGAT
│   │   ├── train.py               <- Training + evaluation
│   │   ├── explain.py             <- Attention -> EvidenceCase -> subgraph
│   │   └── predict.py             <- Confidence-gated routing
│   ├── temporal/
│   │   ├── data.py                <- DailyRecord prefix sequences
│   │   ├── model.py               <- Transformer forecaster + day attention
│   │   ├── train.py               <- Grouped training by patient
│   │   └── predict.py             <- Update-and-forecast API
│   ├── clinical/
│   │   └── daily_update.py        <- Online DailyRecord update + EvidenceCase refresh
│   └── ui/
│       └── app.py                 <- Streamlit implementation
└── scripts/
    ├── benchmark.py
    └── download_embedding_model.py
```

## Thu tu trien khai moi

| # | Viec | File | Uu tien | Trang thai |
|---|---|---|---|---|
| 1 | Cap nhat dependencies | `requirements.txt` | Cao | Can lam |
| 2 | Them Neo4j constraints/indexes | `database.py` | Cao | Can lam |
| 3 | Kiem tra/chuan hoa EvidenceCase + labels + aggregates | `etl_patients.py` | Cao | Da co phan lon |
| 4 | Tao `EvidenceCase -[:MATCHES]-> Diagnostic_Rule` | `etl_patients.py` hoac script moi | Cao | Can lam |
| 5 | Index guideline chunks vao Neo4j | `loader_guidelines.py` | Cao | Can lam |
| 6 | Build tensors + incidence matrix H | `gnn_data.py` | Cao | Can lam |
| 7 | ClinicalHGAT model | `gnn_model.py` | Cao | Can lam |
| 8 | Training loop + evaluation | `gnn_train.py` | Cao | Can lam |
| 9 | Attention -> EvidenceCase -> subgraph | `gnn_explain.py` | Trung binh | Can lam |
| 10 | Confidence-gated routing | `gnn_predict.py` | Trung binh | Can lam |
| 11 | Guideline retrieval + LLM prompt | `rag_engine.py` | Trung binh | Can lam |
| 12 | UI attention heatmap + subgraph panel | `app.py` | Thap | Can lam |

## Yeu cau danh gia hoc thuat

Dataset hien tai chi khoang 148 benh nhan, nen bat buoc danh gia can than:

| Hang muc | Yeu cau |
|---|---|
| Split | Stratified K-Fold, toi thieu 5 folds neu du phan bo label |
| Baseline | Logistic Regression, Random Forest, coverage-score fallback |
| Metrics | AUROC, F1, sensitivity, specificity, calibration |
| Safety | Confidence threshold + fallback |
| Ablation | Khong EvidenceCase, khong Concept, khong guideline RAG |

## Forecasting theo du lieu tung ngay

Ngoai HGAT tren knowledge graph, he thong co them nhanh temporal forecasting de du bao dien bien khi co du lieu ngay moi.

Workflow:

```text
Nhap DailyRecord moi
        |
        v
MERGE DailyRecord + refresh EvidenceCase aggregates
        |
        v
Refresh Concept va MATCHES edges
        |
        v
Temporal Transformer doc chuoi ngay benh
        |
        v
shock_probability + forecast_risk + day_attention
```

Module:

| File | Vai tro |
|---|---|
| `clinical_cdss/clinical/daily_update.py` | Them/cap nhat mot ngay benh va tinh lai EvidenceCase |
| `clinical_cdss/temporal/data.py` | Tao prefix samples: ngay 1..k -> label shock cuoi cung |
| `clinical_cdss/temporal/model.py` | Transformer nhin lai cac ngay cu bang day attention |
| `clinical_cdss/temporal/train.py` | Train theo GroupKFold de khong leak cung benh nhan |
| `clinical_cdss/temporal/predict.py` | `update_and_predict()` cho du lieu ngay moi |

Chay train:

```powershell
python -m clinical_cdss.temporal.train --epochs 120 --out models/temporal_forecaster.pt
```

Dung du bao sau khi nhap ngay moi:

```python
from clinical_cdss.temporal.predict import TemporalProgressionPredictor

predictor = TemporalProgressionPredictor("models/temporal_forecaster.pt")
result = predictor.update_and_predict(
    patient_id="BN001",
    hospital_day=4,
    disease_day=5,
    wbc=2.1,
    plt=68,
    hct=46.5,
    hflc=0.42,
)
```

## Dong gop hoc thuat

1. `EvidenceCase` la hyperedge tuong minh, tao XAI path ro: `Patient -> EvidenceCase -> Rule -> Severity`.
2. 3 tang XAI: HGAT attention cho `where`, graph traversal cho `what`, guideline RAG cho `why`.
3. Inductive inference: benh nhan moi chi can tao node/canh moi va forward lai, khong retrain weights.
4. Confidence-gated safety: fallback sang coverage score khi HGAT khong chac.
5. Guideline grounding: bao cao LLM dua tren text goc BYT, giam suy dien tu do.
