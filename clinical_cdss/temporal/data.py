from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from clinical_cdss.core.database import Neo4jConnection


DAILY_FEATURES = ["disease_day", "wbc", "plt", "hct", "hflc"]
STATIC_FEATURES = ["age", "gender", "weight", "is_child", "admission_day"]

EXCLUDED_CONCEPTS = {
    "Sốc Dengue",
    "Sốc Dengue nặng",
    "Tái sốc",
    "Suy hô hấp",
    "Xuất huyết nặng",
    "Tổn thương tạng",
    "Tổn thương gan nặng / suy gan cấp"
}


@dataclass
class TemporalForecastData:
    sequences: torch.Tensor
    static_features: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    patient_ids: List[str]
    prefix_days: List[int]
    feature_names: List[str]
    static_feature_names: List[str]
    mean: torch.Tensor
    std: torch.Tensor
    static_mean: torch.Tensor
    static_std: torch.Tensor
    symptom_names: List[str]
    safe_concept_names: List[str]


def _to_float(value, default=0.0):
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


def _normalize_with_mask(x: torch.Tensor, mask: torch.Tensor):
    valid = mask.unsqueeze(-1).bool()
    masked = x.masked_fill(~valid, float("nan"))
    mean = torch.nanmean(masked, dim=(0, 1))
    mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
    centered = torch.where(valid, x - mean, torch.zeros_like(x))
    count = valid.sum(dim=(0, 1)).clamp_min(1)
    var = (centered.pow(2).sum(dim=(0, 1)) / count).clamp_min(1e-6)
    std = torch.sqrt(var)
    return torch.where(valid, (x - mean) / std, torch.zeros_like(x)), mean, std


def load_temporal_prefix_data(max_days: int = 10, min_prefix_days: int = 2) -> TemporalForecastData:
    db = Neo4jConnection()
    try:
        # Query patient sequence with symptoms and concepts
        rows = db.execute_query("""
            MATCH (p:Patient)
            WHERE p.binary_label IS NOT NULL
            OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(sym:Symptom)
            WITH p, collect(DISTINCT sym.name) AS symptoms
            OPTIONAL MATCH (p)-[:HAS_CONDITION]->(con:Concept)
            WITH p, symptoms, collect(DISTINCT con.name) AS concepts
            OPTIONAL MATCH (p)-[:HAS_RECORD]->(dr:DailyRecord)
            WITH p, symptoms, concepts, dr
            ORDER BY dr.disease_day, dr.day
            RETURN p.id AS patient_id,
                   p.binary_label AS label,
                   p.age AS age,
                   p.gender AS gender,
                   p.weight AS weight,
                   p.is_child AS is_child,
                   p.admission_day AS admission_day,
                   symptoms,
                   concepts,
                   collect({
                       disease_day: dr.disease_day,
                       wbc: dr.wbc,
                       plt: dr.plt,
                       hct: dr.hct,
                       hflc: dr.hflc
                   }) AS records
            ORDER BY p.id
        """)
    finally:
        db.close()

    # Count frequencies of symptoms and concepts from the queried rows
    symptom_counts = {}
    concept_counts = {}
    for row in rows:
        for sym in (row.get("symptoms") or []):
            symptom_counts[sym] = symptom_counts.get(sym, 0) + 1
        for con in (row.get("concepts") or []):
            concept_counts[con] = concept_counts.get(con, 0) + 1

    # Filter symptoms with frequency >= 5
    symptoms_list = [
        sym for sym, count in symptom_counts.items()
        if count >= 5
    ]
    symptoms_list.sort()

    # Filter safe concepts with frequency >= 20
    safe_concepts_list = [
        con for con, count in concept_counts.items()
        if con not in EXCLUDED_CONCEPTS and count >= 20
    ]
    safe_concepts_list.sort()

    sequences = []
    masks = []
    static_rows = []
    labels = []
    patient_ids = []
    prefix_days = []

    for row in rows:
        records = [r for r in row["records"] if r["disease_day"] is not None]
        records = records[:max_days]
        if len(records) < min_prefix_days:
            continue
        
        # Base static features
        base_static = [_to_float(row[name]) for name in STATIC_FEATURES]
        
        # Symptoms multi-hot flags
        patient_symptoms = set(row.get("symptoms") or [])
        symptom_flags = [1.0 if sym in patient_symptoms else 0.0 for sym in symptoms_list]
        
        # Concepts multi-hot flags
        patient_concepts = set(row.get("concepts") or [])
        concept_flags = [1.0 if con in patient_concepts else 0.0 for con in safe_concepts_list]
        
        # Concat static features
        static = base_static + symptom_flags + concept_flags
        
        for prefix_len in range(min_prefix_days, len(records) + 1):
            prefix = records[:prefix_len]
            seq = torch.zeros((max_days, len(DAILY_FEATURES)), dtype=torch.float32)
            mask = torch.zeros(max_days, dtype=torch.float32)
            for i, rec in enumerate(prefix):
                seq[i] = torch.tensor([_to_float(rec[name]) for name in DAILY_FEATURES])
                mask[i] = 1.0
            sequences.append(seq)
            masks.append(mask)
            static_rows.append(static)
            labels.append(int(row["label"]))
            patient_ids.append(row["patient_id"])
            prefix_days.append(int(prefix[-1]["disease_day"]))

    if not sequences:
        raise ValueError("No temporal training samples found. Load patient DailyRecord data first.")

    seq_tensor = torch.stack(sequences)
    mask_tensor = torch.stack(masks)
    seq_tensor, mean, std = _normalize_with_mask(seq_tensor, mask_tensor)

    static_tensor = torch.tensor(static_rows, dtype=torch.float32)
    static_mean = static_tensor.mean(dim=0)
    static_std = static_tensor.std(dim=0).clamp_min(1e-6)
    static_tensor = (static_tensor - static_mean) / static_std

    return TemporalForecastData(
        sequences=seq_tensor,
        static_features=static_tensor,
        masks=mask_tensor,
        labels=torch.tensor(labels, dtype=torch.long),
        patient_ids=patient_ids,
        prefix_days=prefix_days,
        feature_names=DAILY_FEATURES,
        static_feature_names=STATIC_FEATURES,
        mean=mean,
        std=std,
        static_mean=static_mean,
        static_std=static_std,
        symptom_names=symptoms_list,
        safe_concept_names=safe_concepts_list,
    )


def patient_groups(data: TemporalForecastData) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for idx, patient_id in enumerate(data.patient_ids):
        groups.setdefault(patient_id, []).append(idx)
    return groups
