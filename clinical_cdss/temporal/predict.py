from pathlib import Path
from typing import Optional

import torch

from clinical_cdss.clinical.daily_update import upsert_daily_record
from clinical_cdss.core.database import Neo4jConnection
from clinical_cdss.temporal.model import TemporalDiseaseForecaster


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _to_float(value, default=0.0):
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    return float(value)


class TemporalProgressionPredictor:
    def __init__(self, model_path="models/temporal_forecaster.pt", max_days: int = 10):
        self.model_path = Path(model_path)
        self.max_days = max_days
        checkpoint = _load_checkpoint(self.model_path)
        self.feature_names = checkpoint["feature_names"]
        self.static_feature_names = checkpoint["static_feature_names"]
        self.mean = checkpoint["mean"]
        self.std = checkpoint["std"]
        self.static_mean = checkpoint["static_mean"]
        self.static_std = checkpoint["static_std"]
        self.symptom_names = checkpoint.get("symptom_names", [])
        self.safe_concept_names = checkpoint.get("safe_concept_names", [])
        self.model = TemporalDiseaseForecaster(
            daily_dim=checkpoint["daily_dim"],
            static_dim=checkpoint["static_dim"],
            hidden=checkpoint.get("hidden", 64),
            heads=checkpoint.get("heads", 4),
            layers=checkpoint.get("layers", 2),
            dropout=checkpoint.get("dropout", 0.2),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _patient_sequence(self, patient_id: str):
        db = Neo4jConnection()
        try:
            rows = db.execute_query("""
                MATCH (p:Patient {id: $pid})
                OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(sym:Symptom)
                WITH p, collect(DISTINCT sym.name) AS symptoms
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(con:Concept)
                WITH p, symptoms, collect(DISTINCT con.name) AS concepts
                OPTIONAL MATCH (p)-[:HAS_RECORD]->(dr:DailyRecord)
                WITH p, symptoms, concepts, dr
                ORDER BY dr.disease_day, dr.day
                RETURN p.age AS age,
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
            """, {"pid": patient_id})
        finally:
            db.close()
        if not rows:
            return None
        row = rows[0]
        records = [r for r in row["records"] if r["disease_day"] is not None]
        records = records[: self.max_days]
        seq = torch.zeros((1, self.max_days, len(self.feature_names)), dtype=torch.float32)
        mask = torch.zeros((1, self.max_days), dtype=torch.float32)
        for i, rec in enumerate(records):
            seq[0, i] = torch.tensor([_to_float(rec[name]) for name in self.feature_names])
            mask[0, i] = 1.0
        seq = torch.where(mask.unsqueeze(-1).bool(), (seq - self.mean) / self.std, torch.zeros_like(seq))

        # Base static features
        base_static = [_to_float(row[name]) for name in self.static_feature_names]
        
        # Symptoms multi-hot flags
        patient_symptoms = set(row.get("symptoms") or [])
        symptom_flags = [1.0 if sym in patient_symptoms else 0.0 for sym in self.symptom_names]
        
        # Concepts multi-hot flags
        patient_concepts = set(row.get("concepts") or [])
        concept_flags = [1.0 if con in patient_concepts else 0.0 for con in self.safe_concept_names]
        
        # Concat static features
        static_list = base_static + symptom_flags + concept_flags
        
        static = torch.tensor([static_list], dtype=torch.float32)
        static = (static - self.static_mean) / self.static_std
        return seq, static, mask, records

    def predict(self, patient_id: str):
        payload = self._patient_sequence(patient_id)
        if payload is None:
            return {"error": f"Patient not found: {patient_id}"}
        seq, static, mask, records = payload
        if not records:
            return {"error": f"No DailyRecord found for patient {patient_id}"}

        with torch.no_grad():
            output = self.model(seq, static, mask)
            prob = torch.softmax(output["logits"], dim=-1)[0]
            attention = output["day_attention"][0].cpu().tolist()

        day_attention = []
        for idx, rec in enumerate(records):
            day_attention.append({
                "disease_day": rec["disease_day"],
                "hospital_day_index": idx + 1,
                "attention": attention[idx],
                "wbc": rec["wbc"],
                "plt": rec["plt"],
                "hct": rec["hct"],
                "hflc": rec["hflc"],
            })

        return {
            "patient_id": patient_id,
            "shock_probability": float(prob[1]),
            "non_shock_probability": float(prob[0]),
            "forecast_risk": float(output["forecast_risk"][0]),
            "day_attention": day_attention,
        }

    def update_and_predict(
        self,
        patient_id: str,
        hospital_day: int,
        disease_day: Optional[int] = None,
        wbc: Optional[float] = None,
        plt: Optional[float] = None,
        hct: Optional[float] = None,
        hflc: Optional[float] = None,
    ):
        update_summary = upsert_daily_record(
            patient_id=patient_id,
            hospital_day=hospital_day,
            disease_day=disease_day,
            wbc=wbc,
            plt=plt,
            hct=hct,
            hflc=hflc,
        )
        if "error" in update_summary:
            return update_summary
        prediction = self.predict(patient_id)
        prediction["update_summary"] = update_summary
        return prediction
