from pathlib import Path
from typing import Dict, Optional

import torch

from clinical_cdss.gnn.data import load_graph_data
from clinical_cdss.gnn.explain import retrieve_subgraph, top_attention_nodes, top_evidence_case
from clinical_cdss.gnn.model import ClinicalHGAT
from clinical_cdss.rag.engine import MedicalGraphRAG


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


class ClinicalCDSS:
    def __init__(
        self,
        model_path="models/clinical_hgat.pt",
        confidence_threshold: float = 0.75,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.rag = MedicalGraphRAG()
        self.model = None
        self.data = None
        if self.model_path.exists():
            self._load_model()

    def _load_model(self):
        checkpoint = _load_checkpoint(self.model_path)
        self.data = load_graph_data()
        # Check dimension compatibility between checkpoint and current graph
        current_dims = {k: v.shape[1] for k, v in self.data.x_dict.items()}
        saved_dims = checkpoint.get("in_dim_dict", {})
        mismatches = {k: (saved_dims[k], current_dims.get(k)) for k in saved_dims
                      if current_dims.get(k) != saved_dims[k]}
        if mismatches:
            print(f"[HGAT] Dimension mismatch detected: {mismatches}")
            print("[HGAT] Model disabled — retrain with: python -m clinical_cdss.gnn.train")
            self.model = None
            self._dim_mismatch = mismatches
            return
        self._dim_mismatch = None
        self.model = ClinicalHGAT(saved_dims, num_classes=3)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _hgat_predict(self, patient_id: str) -> Optional[Dict]:
        if self.model is None or self.data is None:
            return None
        if patient_id not in self.data.patient_ids:
            return None

        patient_idx = self.data.patient_ids.index(patient_id)
        try:
            with torch.no_grad():
                output = self.model(self.data)
                logits = output["logits"][patient_idx]
                prob = torch.softmax(logits, dim=-1)
                diagnosis = int(prob.argmax().item())
                confidence = float(prob.max().item())
        except RuntimeError as exc:
            print(f"[HGAT] Inference error: {exc}")
            return None

        evidence_id, evidence_weight = top_evidence_case(self.data, output, patient_idx)
        subgraph = retrieve_subgraph(patient_id, evidence_id)
        if evidence_id:
            subgraph["attention_nodes"] = top_attention_nodes(self.data, output, evidence_id)
            subgraph["evidence_attention"] = evidence_weight

        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "subgraph": subgraph,
        }

    def diagnose(self, patient_id: str, use_llm: bool = True) -> Dict:
        hgat_result = self._hgat_predict(patient_id)
        
        # Kiểm tra điều kiện định tuyến an toàn (GNN không được đè ca bệnh nặng thành nhẹ)
        use_gnn = False
        if hgat_result and hgat_result["confidence"] >= self.confidence_threshold:
            subgraph = hgat_result["subgraph"]
            severity = str(subgraph.get("severity") or "").lower()
            concepts = [str(c).lower() for c in subgraph.get("concepts", [])]
            
            # Các dấu hiệu/luật cảnh báo mức độ nặng
            severe_keywords = ["sốc", "nặng", "tái sốc", "suy gan", "suy tạng", "suy hô hấp", "xuất huyết nặng", "phù phổi", "suy tim", "viêm cơ tim", "thể não"]
            
            def is_severe(text: str) -> bool:
                t = text.lower()
                if "cảnh báo sốc" in t or "giảm tiểu cầu nặng" in t or "giảm tiểu cầu rất nặng" in t:
                    return False
                return any(kw in t for kw in severe_keywords)
                
            has_severe_trigger = is_severe(severity) or any(is_severe(c) for c in concepts)
            
            # Nếu có dấu hiệu nặng nhưng GNN lại chẩn đoán là nhẹ hoặc cảnh báo (< 2), ép buộc fallback để an toàn lâm sàng
            if has_severe_trigger and hgat_result["diagnosis"] < 2:
                use_gnn = False
            else:
                use_gnn = True

        if use_gnn:
            subgraph = hgat_result["subgraph"]
            guideline = self.rag.retrieve_guideline_chunk(
                subgraph.get("matched_rule") or "",
                subgraph.get("concepts", []),
            )
            report = self.rag.generate_response_from_subgraph(
                patient_id=patient_id,
                subgraph=subgraph,
                guideline_chunk=guideline,
                method=f"HGAT (conf={hgat_result['confidence']:.0%})",
                use_llm=use_llm,
            )
            return {
                "diagnosis": hgat_result["diagnosis"],
                "confidence": hgat_result["confidence"],
                "method": "HGAT",
                "subgraph": subgraph,
                "report": report,
            }

        context = self.rag.retrieve_context(patient_id)
        if "error" in context:
            return context
        guideline = self.rag.retrieve_guideline_chunk(
            context.get("rule_name", ""),
            context.get("concepts", []),
        )
        report = self.rag.generate_response(
            context,
            guideline_chunk=guideline,
            use_llm=use_llm,
        )
        # Xác định chẩn đoán dự phòng an toàn (diagnosis fallback)
        fb_severity = str(context.get("severity") or "").lower()
        fb_concepts = [str(c).lower() for c in context.get("concepts", [])]
        
        severe_keywords = ["sốc", "nặng", "tái sốc", "suy gan", "suy tạng", "suy hô hấp", "xuất huyết nặng", "phù phổi", "suy tim", "viêm cơ tim", "thể não"]
        def is_severe_fb(t: str) -> bool:
            if "cảnh báo sốc" in t or "giảm tiểu cầu nặng" in t or "giảm tiểu cầu rất nặng" in t:
                return False
            return any(kw in t for kw in severe_keywords)
            
        has_severe_fb = is_severe_fb(fb_severity) or any(is_severe_fb(c) for c in fb_concepts)
        
        warning_keywords = ["cảnh báo", "chuyển độ", "cô đặc máu", "giảm tiểu cầu", "thoát huyết tương", "đau bụng", "nôn", "gan to", "mệt mỏi", "lừ đừ", "li bì", "vật vã", "xuất huyết"]
        has_warning_fb = any(kw in fb_severity for kw in ["cảnh báo", "chuyển độ"]) or \
                         any(any(kw in c for kw in warning_keywords) for c in fb_concepts)
                         
        if has_severe_fb:
            diagnosis = 2
        elif has_warning_fb:
            diagnosis = 1
        else:
            diagnosis = 0
            
        confidence = hgat_result["confidence"] if hgat_result else 0.0
        return {
            "diagnosis": diagnosis,
            "confidence": confidence,
            "method": "Coverage Score fallback",
            "subgraph": context,
            "report": report,
        }
