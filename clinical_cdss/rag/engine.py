import os
import re
from pathlib import Path
from clinical_cdss.core.database import Neo4jConnection
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parents[2]
EMBEDDER = None


def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        EMBEDDER = SentenceTransformer(str(BASE_DIR / "local_models" / "vietnamese-sbert"))
    return EMBEDDER

class MedicalGraphRAG:
    def __init__(self):
        cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)

        self.db = Neo4jConnection()
        self.llm = None

    def _get_llm(self):
        if self.llm is not None:
            return self.llm

        model_path = str(BASE_DIR / "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_threads=4,
                verbose=False
            )
        except Exception as exc:
            print(f"GPU LLM load failed, falling back to CPU: {exc}")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=0,
                n_threads=4,
                verbose=False
            )
        return self.llm

    def close(self):
        self.db.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _generate_with_llm(self, prompt: str):
        llm = self._get_llm()
        return llm(
            prompt,
            max_tokens=1024,
            temperature=0.0,
            stop=["Assistant:", "Chúc bạn"],
        )

    def _clean_output(self, text: str) -> str:
        if not text: return "Chưa đủ dữ liệu."
        
        # Truncate any extra sections generated after Part 3 to keep only the requested 3 sections
        idx3 = text.find("**3. ")
        if idx3 == -1:
            idx3 = text.find("**3.")
        if idx3 != -1:
            # Look for the next header starting on a new line (e.g. \n**) to skip the closing ** of the Part 3 title
            next_header = text.find("\n**", idx3 + 10)
            if next_header != -1:
                text = text[:next_header].strip()

        stop_markers = [
            "Assistant:", "Chúc bạn", "Yêu bạn", "😊", "❤️", "(Phần trên",
            "Báo cáo đã được cập nhật", "Dưới đây là báo cáo đầy đủ",
            "Báo cáo được cập nhật", "Hy vọng báo cáo",
        ]
        for marker in stop_markers:
            if marker in text: text = text.split(marker)[0]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = []
        seen = set()
        for line in lines:
            if line.lower() in {"trả lời ngắn gọn, rõ ràng, không lập.", "chỉ trả lời 1 lần duy nhất"}: continue
            key = re.sub(r"\s+", " ", line).strip().lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(line)
        return "\n".join(cleaned).strip() or "Chưa đủ dữ liệu."

    def _remove_bullets(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("- ", "• ")):
                stripped = stripped[2:].strip()
            lines.append(stripped)
        return "\n".join(lines).strip()


    def retrieve_context(self, patient_name: str) -> dict:
        # ===================================================================
        # THUẬT TOÁN HYPERGRAPH COVERAGE SCORE ĐA CHIỀU (3-Dimension)
        # Siêu cạnh (Hyper-edge) = Diagnostic_Rule kết nối đồng thời:
        #   - Chiều 1: Symptom Nodes    (W=0.5) - Triệu chứng lâm sàng
        #   - Chiều 2: LabTest Nodes    (W=0.3) - Chỉ số cận lâm sàng
        #   - Chiều 3: Concept Nodes    (W=0.2) - Khái niệm rủi ro suy diễn
        # ===================================================================
        query = """
        // --- Chiều 1: Lấy Symptom của bệnh nhân ---
        MATCH (p:Patient {id: $patient_name})
        OPTIONAL MATCH (p)-[:HAS_SYMPTOM]->(s:Symptom)
        WITH p, collect(DISTINCT s) AS patient_symptoms
        
        // --- Chiều 2: Lấy Concept (Khái niệm suy diễn: Cô đặc máu, Giảm tiểu cầu) ---
        OPTIONAL MATCH (p)-[:HAS_CONDITION]->(con:Concept)
        WITH p, patient_symptoms, collect(DISTINCT con.name) AS patient_concepts

        // --- Tìm Phác đồ và tính Coverage Score từng chiều ---
        MATCH (rule:Diagnostic_Rule)
        
        // Chiều 1: Symptom match
        OPTIONAL MATCH (rule)<-[:PART_OF_RULE]-(rs:Symptom)
        WITH p, patient_symptoms, patient_concepts, rule, collect(DISTINCT rs) AS rule_symptoms

        // Chiều 2: LabTest match qua Semantic Bridge (LabTest -[:MAPS_TO]-> Concept <- bệnh nhân)
        OPTIONAL MATCH (rule)<-[:PART_OF_RULE]-(lt:LabTest)-[:MAPS_TO]->(mapped_concept:Concept)<-[:HAS_CONDITION]-(p)
        WITH p, patient_symptoms, patient_concepts, rule, rule_symptoms, count(DISTINCT lt) AS lab_match
        
        OPTIONAL MATCH (rule)<-[:PART_OF_RULE]-(all_lt:LabTest)
        WITH p, patient_symptoms, patient_concepts, rule, rule_symptoms, lab_match, count(DISTINCT all_lt) AS lab_total

        // Chiều 3: Concept match trực tiếp
        OPTIONAL MATCH (rule)<-[:LINKED_TO_RULE]-(rc:Concept)
        WHERE rc.name IN patient_concepts
        WITH p, patient_symptoms, patient_concepts, rule, rule_symptoms,
             lab_match, lab_total, collect(DISTINCT rc.name) AS matched_concept_names

        OPTIONAL MATCH (rule)<-[:LINKED_TO_RULE]-(all_rc:Concept)
        WITH rule, rule_symptoms, patient_symptoms, patient_concepts, lab_match, lab_total, matched_concept_names,
             collect(DISTINCT all_rc.name) AS rule_concept_names, count(DISTINCT all_rc) AS concept_total
        WHERE ALL(cname IN rule_concept_names WHERE NOT cname IN ['Bệnh nhi', 'Người lớn', 'Nhũ nhi', 'Thai kỳ'] OR cname IN patient_concepts)

        // Tính số lượng Concept lâm sàng khớp (loại trừ các concept ngữ cảnh/phi lâm sàng)
        WITH rule, rule_symptoms, patient_symptoms, lab_match, lab_total, concept_total, matched_concept_names,
             size([c IN matched_concept_names WHERE NOT c IN [
                 'Giai đoạn nguy hiểm theo ngày bệnh', 'Giai đoạn sốt', 'Giai đoạn hồi phục', 
                 'Giai đoạn sớm', 'Giai đoạn chuyển tiếp nguy hiểm/hồi phục', 
                 'Bệnh nhi', 'Người lớn', 'Nhũ nhi', 'Thai kỳ'
             ]]) AS clinical_concept_match,
             size(matched_concept_names) AS concept_match

        // Tính số lượng Triệu chứng khớp (hỗ trợ so khớp mềm qua Concept chung)
        WITH rule, concept_match, concept_total, clinical_concept_match, lab_match, lab_total,
             size([rs IN rule_symptoms WHERE rs IN patient_symptoms OR EXISTS {
                 MATCH (rs)-[:MAPS_TO]->(c:Concept)<-[:MAPS_TO]-(ps:Symptom)
                 WHERE ps IN patient_symptoms
             }]) AS sym_match,
             size(rule_symptoms) AS sym_total

        // --- Tính Điểm Weighted Hypergraph Coverage Score ---
        WITH rule, sym_match, sym_total, lab_match, lab_total, concept_match, concept_total, clinical_concept_match,
             CASE WHEN sym_total > 0  THEN toFloat(sym_match) / toFloat(sym_total) ELSE 0.0 END AS sym_score,
             CASE WHEN lab_total > 0 THEN toFloat(lab_match) / toFloat(lab_total) ELSE 0.0 END AS lab_score,
             CASE WHEN concept_total > 0 THEN toFloat(concept_match) / toFloat(concept_total) ELSE 0.0 END AS concept_score
        
        // Trọng số 50% Symptom, 30% LabTest, 20% Concept
        WITH rule, sym_match, sym_total, lab_match, lab_total, concept_match, concept_total, clinical_concept_match,
             (0.5 * sym_score) + (0.3 * lab_score) + (0.2 * concept_score) AS coverage_score
             
        WHERE coverage_score >= 0.15
          // Bộ lọc: loại bỏ luật rỗng hoặc chỉ khớp thông tin ngữ cảnh thời gian/tuổi
          AND NOT (sym_match = 0 AND clinical_concept_match = 0)
        
        OPTIONAL MATCH (rule)-[:DETERMINES]->(sev:Severity)
        
        RETURN rule.name AS rule_name,
               sym_match, sym_total, lab_match, lab_total, concept_match, concept_total,
               rule.phase AS phase,
               sev.name AS severity,
               coverage_score
         ORDER BY coverage_score DESC,
                  CASE
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốc' AND (coalesce(toLower(sev.name), '') CONTAINS 'nặng' OR coalesce(toLower(sev.name), '') CONTAINS 'kéo dài' OR coalesce(toLower(sev.name), '') CONTAINS 'thất bại') THEN 5
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốc' THEN 4
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'cảnh báo' OR coalesce(toLower(sev.name), '') CONTAINS 'chuyển độ' THEN 3
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốt xuất huyết' OR coalesce(toLower(sev.name), '') CONTAINS 'sốt dengue nặng' THEN 2
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'sốt' OR coalesce(toLower(sev.name), '') CONTAINS 'chăm sóc' OR coalesce(toLower(sev.name), '') CONTAINS 'lo lắng' THEN 1
                       WHEN coalesce(toLower(sev.name), '') CONTAINS 'chưa phân loại' THEN 0
                       ELSE 1 END DESC,
                  rule_name ASC
         LIMIT 3
        """
        result = self.db.execute_query(query, {"patient_name": patient_name})

        patient_rows = self.db.execute_query("""
            MATCH (p:Patient {id: $patient_name})
            OPTIONAL MATCH (p)-[:HAS_EVIDENCE]->(ec:EvidenceCase)
            OPTIONAL MATCH (p)-[:HAS_RECORD]->(dr:DailyRecord)
            WITH p, ec, max(dr.disease_day) AS current_disease_day
            RETURN p.age AS age,
                   p.gender AS gender,
                   p.is_child AS is_child,
                   p.admission_day AS admission_day,
                   current_disease_day,
                   coalesce(ec.ast_value, p.ast_value) AS ast_value,
                   coalesce(ec.alt_value, p.alt_value) AS alt_value,
                   coalesce(ec.creatinine_value, p.creatinine_value) AS creatinine_value,
                   coalesce(ec.plt_nadir, p.plt_nadir) AS plt_nadir,
                   coalesce(ec.hct_peak, p.hct_peak) AS hct_peak,
                   coalesce(ec.hflc_peak, p.hflc_peak) AS hflc_peak
        """, {"patient_name": patient_name})
        patient_info = dict(patient_rows[0]) if patient_rows else {}
        if not patient_info:
            return {"error": f"⚠️ Không tìm thấy bệnh nhân có ID/Tên '{patient_name}' trong hệ thống."}

        if not result:
            # Fallback: Lấy Symptom thuần túy nếu không có phác đồ nào đạt ngưỡng
            fallback_query = """
            MATCH (p:Patient {id: $patient_name})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN collect(DISTINCT s.name) AS all_symptoms
            """
            fb_res = self.db.execute_query(fallback_query, {"patient_name": patient_name})
            symptoms = fb_res[0]["all_symptoms"] if fb_res and fb_res[0]["all_symptoms"] else []
            rule_name = ""
            phase_text = "Chưa rõ"
            severity_text = "Chưa xác định"
            explainability_text = "Không tìm thấy đường đi (Path) nào trên đồ thị thỏa mãn ngưỡng 15% (3-chiều)."
        else:
            best_match = result[0]
            score = round(best_match["coverage_score"] * 100, 1)
            rule_name = best_match["rule_name"] if best_match["rule_name"] else "Phác đồ Chung"
            phase_text = best_match["phase"] if best_match["phase"] else "Chưa rõ"
            severity_text = best_match["severity"] if best_match["severity"] else "Chưa xác định"

            sym_match = best_match["sym_match"]
            sym_total = best_match["sym_total"]
            lab_match = best_match["lab_match"]
            lab_total = best_match["lab_total"]
            concept_match = best_match["concept_match"]
            concept_total = best_match["concept_total"]

            # Lấy danh sách triệu chứng bệnh nhân riêng (query phụ)
            sym_query = """
            MATCH (p:Patient {id: $patient_name})-[:HAS_SYMPTOM]->(s:Symptom)
            RETURN collect(DISTINCT s.name) AS all_symptoms
            """
            sym_res = self.db.execute_query(sym_query, {"patient_name": patient_name})
            symptoms = sym_res[0]["all_symptoms"] if sym_res else []

            # Explainability: giải thích đầy đủ 3 chiều
            explainability_text = (
                f"Phác đồ chọn: [{rule_name}] | Hypergraph Coverage Score: {score}%\n"
                f"  - Chiều Triệu chứng (W=0.5): {sym_match}/{sym_total} Node [Symptom] khớp\n"
                f"  - Chiều Cận lâm sàng (W=0.3): {lab_match}/{lab_total} Node [LabTest] khớp qua Concept\n"
                f"  - Chiều Khái niệm (W=0.2): {concept_match}/{concept_total} Node [Concept] khớp trực tiếp\n"
                f"  - Cơ chế: Patient → Symptom/Concept → Hyper-edge → Diagnostic_Rule"
            )

        symptoms_text = ", ".join(symptoms) if symptoms else "Không ghi nhận"

        # CÂU LỆNH CYPHER MỚI: Lấy Dữ liệu Chuỗi thời gian (Ngày 1 - Ngày 10)
        query_records = """
        MATCH (p:Patient {id: $patient_name})-[:HAS_RECORD]->(dr:DailyRecord)
        RETURN dr.day AS day, dr.wbc AS wbc, dr.plt AS plt, dr.hct AS hct, dr.hflc AS hflc
        ORDER BY dr.day ASC
        """
        records_result = self.db.execute_query(query_records, {"patient_name": patient_name})

        timeline_text = ""
        if records_result:
            timeline_text = "Diễn tiến Cận lâm sàng (BC: Bạch cầu, TC: Tiểu cầu, HCT: Dung tích HC):\n"
            for rec in records_result:
                day = rec["day"]
                wbc = rec["wbc"] if rec["wbc"] is not None else "-"
                plt = rec["plt"] if rec["plt"] is not None else "-"
                hct = rec["hct"] if rec["hct"] is not None else "-"
                hflc = rec["hflc"] if rec["hflc"] is not None else "-"
                timeline_text += f"- Ngày {day}: BC: {wbc}, TC: {plt}, HCT: {hct}, HFLC: {hflc}\n"
        else:
            timeline_text = "Không có dữ liệu cận lâm sàng theo ngày.\n"

        # CÂU LỆNH CYPHER MỚI: Lấy Khái niệm Y khoa (Concept Mapping)
        query_concepts = """
        MATCH (p:Patient {id: $patient_name})-[:HAS_CONDITION]->(c:Concept)
        RETURN collect(DISTINCT c.name) AS concepts
        """
        concept_result = self.db.execute_query(query_concepts, {"patient_name": patient_name})
        concepts = concept_result[0]["concepts"] if concept_result and concept_result[0]["concepts"] else []
        concepts_text = ", ".join(concepts) if concepts else "Chưa ghi nhận biến chứng cận lâm sàng."

        age = patient_info.get("age")
        current_disease_day = patient_info.get("current_disease_day")
        ast_value = patient_info.get("ast_value") or 0.0
        alt_value = patient_info.get("alt_value") or 0.0
        creatinine_value = patient_info.get("creatinine_value") or 0.0
        is_child = bool(patient_info.get("is_child"))

        organ_flags = []
        if ast_value >= 1000 or alt_value >= 1000:
            organ_flags.append(
                "AST/ALT >= 1000 U/L: đủ tiêu chuẩn gợi ý SXH Dengue nặng do tổn thương tạng theo phác đồ."
            )
        if creatinine_value >= (70.0 if is_child else 110.0):
            organ_flags.append(
                "Creatinine tăng so với ngưỡng tham chiếu theo nhóm tuổi: gợi ý tổn thương thận cấp hoặc giảm tưới máu tạng."
            )
        organ_assessment_text = "\n".join(f"- {item}" for item in organ_flags) if organ_flags else "- Chưa ghi nhận tiêu chí tổn thương tạng rõ từ AST/ALT/Creatinine."
        demographics_text = (
            f"Tuổi: {age if age is not None else 'không rõ'}; "
            f"nhóm tuổi: {'nhi' if is_child else 'người lớn'}; "
            f"ngày bệnh hiện tại: {current_disease_day if current_disease_day is not None else 'không rõ'}"
        )
        organ_labs_text = (
            f"AST={ast_value if ast_value else 'không có'}, "
            f"ALT={alt_value if alt_value else 'không có'}, "
            f"Creatinine={creatinine_value if creatinine_value else 'không có'} umol/L"
        )

        plt_nadir = patient_info.get("plt_nadir")
        hct_peak = patient_info.get("hct_peak")
        hflc_peak = patient_info.get("hflc_peak")
        hematology_summary = (
            f"Tiểu cầu thấp nhất (PLT nadir): {plt_nadir if plt_nadir is not None else 'không có'} G/L; "
            f"Cô đặc máu cao nhất (HCT peak): {hct_peak if hct_peak is not None else 'không có'}%; "
            f"Chỉ số HFLC cao nhất (HFLC peak): {hflc_peak if hflc_peak is not None else 'không có'}%"
        )

        context_str = f"""
Bệnh nhân: {patient_name}
Thông tin nền: {demographics_text}
Triệu chứng hiện tại: {symptoms_text}
Chỉ số huyết học tổng hợp: {hematology_summary}
{timeline_text}
Men gan / chức năng thận: {organ_labs_text}
Đánh giá tổn thương tạng:
{organ_assessment_text}
Khái niệm y khoa: {concepts_text}
Phác đồ phù hợp nhất: {phase_text} | Mức độ: {severity_text}
"""
        return {
            "patient_name": patient_name,
            "symptoms_text": symptoms_text,
            "timeline_text": timeline_text,
            "phase_text": phase_text,
            "severity_text": severity_text,
            "severity": severity_text,
            "demographics_text": demographics_text,
            "organ_labs_text": organ_labs_text,
            "organ_assessment_text": organ_assessment_text,
            "age": age,
            "current_disease_day": current_disease_day,
            "ast_value": ast_value,
            "alt_value": alt_value,
            "creatinine_value": creatinine_value,
            "plt_nadir": plt_nadir,
            "hct_peak": hct_peak,
            "hflc_peak": hflc_peak,
            "concepts_text": concepts_text,
            "concepts": concepts,
            "rule_name": rule_name,
            "explainability_text": explainability_text,
            "coverage_score": result[0]["coverage_score"] if result else 0.0,
            "context_str": context_str,
        }

    def retrieve_guideline_chunk(self, rule_name: str, concepts: list, top_k: int = 3) -> str:
        query_text = " ".join([rule_name or "", *[str(c) for c in concepts if c]])
        if not query_text.strip():
            return ""
        vector = get_embedder().encode(query_text).tolist()
        try:
            results = self.db.execute_query("""
                CALL db.index.vector.queryNodes('guideline_chunk_index', $top_k, $vector)
                YIELD node AS chunk, score
                WHERE score > 0.75
                RETURN chunk.text AS text, chunk.source AS source,
                       chunk.page_start AS page_start, score
                ORDER BY score DESC
            """, {"vector": vector, "top_k": top_k})
        except Exception:
            return ""
        return "\n".join(
            f"[{r['source']} p.{r['page_start']} score={float(r['score']):.2f}] {r['text']}"
            for r in results
        )

    @staticmethod
    def _parse_val(text: str, key: str):
        """Extract first numeric value matching 'KEY=number' from text."""
        m = re.search(key + r"[=:\s]*([\d.]+)", text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def _structured_response(self, context_dict: dict) -> str:
        demo       = context_dict.get("demographics_text", "Bệnh nhân")
        symptoms   = context_dict.get("symptoms_text", "không ghi nhận")
        timeline   = self._remove_bullets(context_dict.get("timeline_text", ""))
        organ_labs = context_dict.get("organ_labs_text", "")
        organ_asm  = context_dict.get("organ_assessment_text", "")
        concepts   = context_dict.get("concepts_text", "")
        severity   = context_dict.get("severity_text", "chưa xác định")
        phase      = context_dict.get("phase_text", "chưa rõ")
        xai_raw    = context_dict.get("explainability_text", "")

        # ── Fix: positive organ damage only (exclude "Chưa ghi nhận") ────
        has_organ_damage = (
            any(kw in organ_asm for kw in ("≥ 1000", "vượt ngưỡng", "thận cấp", "suy gan"))
            and "chưa ghi nhận" not in organ_asm.lower()
        )

        # ── Directly fetch key lab values from context_dict ──────────────
        plt_val = context_dict.get("plt_nadir")
        hct_val = context_dict.get("hct_peak")
        ast_val = context_dict.get("ast_value")
        alt_val = context_dict.get("alt_value")

        # ── Interpret PLT clinically ──────────────────────────────────────
        def plt_desc(v):
            if v is None: return None
            if v < 10:  return f"tiểu cầu giảm rất nặng (PLT={v:.0f} G/L, dưới ngưỡng 10), nguy cơ xuất huyết nặng cực cao"
            if v < 20:  return f"tiểu cầu giảm nặng (PLT={v:.0f} G/L, dưới ngưỡng 20)"
            if v < 50:  return f"tiểu cầu giảm đáng kể (PLT={v:.0f} G/L)"
            if v < 100: return f"tiểu cầu giảm vừa (PLT={v:.0f} G/L)"
            return None

        def hct_desc(v):
            if v is None: return None
            if v > 50:  return f"cô đặc máu nặng (HCT={v:.1f}%, vượt ngưỡng 50%)"
            if v > 44:  return f"cô đặc máu (HCT={v:.1f}%, tăng so với bình thường)"
            return None

        # ── Concept interpretation (no raw list dump) ─────────────────────
        c_list = [c.strip().lower() for c in concepts.split(",") if c.strip()]
        has_shock_warning  = any("sốc" in c or "cảnh báo sốc" in c for c in c_list)
        has_danger_phase   = any("giai đoạn nguy hiểm" in c for c in c_list)
        has_hemorrhage     = any("xuất huyết" in c for c in c_list)

        # ── SECTION 1 ─────────────────────────────────────────────────────
        s1_lines = [f"Bệnh nhân {demo}"]

        if symptoms and symptoms.lower() != "không ghi nhận":
            s1_lines.append(f"nhập viện với biểu hiện {symptoms.lower()}")

        # Lab interpretation
        lab_findings = []
        pd = plt_desc(plt_val)
        hd = hct_desc(hct_val)
        if pd: lab_findings.append(pd)
        if hd: lab_findings.append(hd)
        if lab_findings:
            s1_lines.append("Xét nghiệm ghi nhận " + " và ".join(lab_findings))

        # Organ damage (only if confirmed positive)
        if has_organ_damage:
            clean = organ_asm.replace("- ", "").strip()
            s1_lines.append(f"Đặc biệt: {clean}")
        elif ast_val and ast_val > 0:
            s1_lines.append(
                f"Men gan AST={ast_val:.0f} U/L"
                + (" — chưa đạt ngưỡng tổn thương tạng (< 1000 U/L)" if ast_val < 1000 else "")
            )

        # Risk context
        risk_parts = []
        if has_shock_warning:   risk_parts.append("dấu hiệu cảnh báo sốc Dengue")
        if has_danger_phase:    risk_parts.append("đang trong giai đoạn nguy hiểm của bệnh")
        if has_hemorrhage:      risk_parts.append("có biểu hiện xuất huyết"  )
        if risk_parts:
            s1_lines.append("Bối cảnh lâm sàng: " + ", ".join(risk_parts))

        # Timeline trend
        tl_lines = [l for l in timeline.splitlines()
                    if l.strip() and "chưa có" not in l.lower() and "chỉ số" not in l.lower()]
        if len(tl_lines) >= 3:
            trend_desc = []
            if plt_val is not None:
                trend_desc.append(f"tiểu cầu giảm xuống mức thấp nhất là {plt_val:.0f} G/L")
            if hct_val is not None:
                trend_desc.append(f"cô đặc máu đạt mức cao nhất là {hct_val:.1f}%")
            
            if trend_desc:
                s1_lines.append(
                    f"Theo dõi sát chuỗi xét nghiệm trong {len(tl_lines)} ngày ghi nhận "
                    f"{' và '.join(trend_desc)}"
                )
            else:
                s1_lines.append(
                    f"Chuỗi xét nghiệm theo dõi trong {len(tl_lines)} ngày cần được giám sát chặt chẽ"
                )

        s1 = "**1. Đánh giá Lâm sàng & Cận lâm sàng:**\n" + ". ".join(s1_lines) + "."

        # ── SECTION 2 ─────────────────────────────────────────────────────
        if has_organ_damage:
            s2_body = (
                f"Bệnh nhân đủ tiêu chuẩn phân độ **SXH Dengue nặng** (thể tổn thương tạng) "
                f"theo Phác đồ BYT 2019. Mức tổn thương tạng được ghi nhận vượt ngưỡng "
                f"lâm sàng, đặt bệnh nhân vào nhóm nguy cơ cao hơn nhóm chỉ có dấu hiệu cảnh báo. "
                f"Phác đồ tham chiếu: {phase}."
            )
        else:
            # Build specific reasoning from lab values
            reasoning = []
            if plt_val is not None and plt_val < 20:
                reasoning.append(
                    f"tiểu cầu ở mức nguy hiểm (PLT={plt_val:.0f} G/L) là yếu tố trọng tâm"
                )
            if hct_val is not None and hct_val > 44:
                reasoning.append(
                    f"Hematocrit tăng ({hct_val:.1f}%) phản ánh tình trạng cô đặc máu"
                )
            if ast_val is not None and ast_val < 1000:
                reasoning.append(
                    f"AST={ast_val:.0f} U/L chưa đạt ngưỡng tổn thương tạng"
                )
            if has_shock_warning and not has_organ_damage:
                reasoning.append(
                    "các dấu hiệu cảnh báo sốc hiện diện nhưng chưa đủ tiêu chí sốc thực sự"
                )

            reason_str = "; ".join(reasoning) if reasoning else "dữ liệu lâm sàng hiện có"
            s2_body = (
                f"Tổng hợp lâm sàng phân loại bệnh nhân vào nhóm **{severity}** — "
                f"phác đồ tham chiếu: {phase}. "
                f"Cơ sở phân loại: {reason_str}. "
                f"Chưa đủ tiêu chí phân độ SXH Dengue nặng (AST/ALT < 1000 U/L, "
                f"chưa có bằng chứng suy tạng), tuy nhiên tình trạng tiểu cầu và "
                f"Hematocrit đòi hỏi theo dõi sát tại cơ sở có khả năng can thiệp."
            )
        s2 = "**2. Giai đoạn & Mức độ:**\n" + s2_body

        # ── SECTION 3: XAI in clinical language ──────────────────────────
        # Translate raw XAI text into plain clinical explanation
        xai_clean = " ".join(self._remove_bullets(xai_raw).split())
        # Split on "; " separators, extract (XX%) from the END of each segment
        xai_nodes_part = xai_clean.split("Phác đồ")[0]  # stop before "Phác đồ"
        node_matches = []
        for seg in xai_nodes_part.split(";"):
            seg = seg.strip().lstrip("Hệ thống gán trọng số cao nhất cho các yếu tố:").strip()
            last_pct = re.search(r"\((\d+)%\)\s*$", seg)
            if last_pct:
                pct  = last_pct.group(1)
                name = seg[:last_pct.start()].strip()
                # Strip trailing sub-type qualifier like "(Sốc Dengue)"
                name = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
                if name:
                    node_matches.append((name, pct))
        if node_matches:
            top = node_matches[:3]
            cov_match = re.search(r"Coverage (\d+%)", xai_clean)
            cov_str = cov_match.group(1) if cov_match else "N/A"

            # Detect nodes that contradict confirmed patient data
            _organ_kws = {"tổn thương gan", "suy gan", "tổn thương tạng", "tổn thương thận"}
            node_lines = []
            for n, w in top:
                n_lower = n.lower()
                is_organ = any(kw in n_lower for kw in _organ_kws)
                if is_organ and not has_organ_damage:
                    node_lines.append(
                        f"**{n}** — nút đồ thị liên kết cấu trúc với phác đồ {phase} "
                        f"(attention {w}%; *bệnh nhân này chưa xác nhận tổn thương tạng*)"
                    )
                else:
                    node_lines.append(f"**{n}** (trọng số {w}%)")

            xai_explain = (
                f"Mô hình HGAT phân loại **{phase}** dựa trên đường dẫn suy luận đồ thị "
                f"(độ phủ {cov_str}). Các nút có trọng số attention cao nhất:\n"
                + "\n".join(f"- {line}" for line in node_lines)
                + "\n\n*Lưu ý: Trọng số attention phản ánh cấu trúc kết nối đồ thị lâm sàng — "
                "không đồng nghĩa với việc bệnh nhân được xác nhận có các tình trạng đó.*"
            )
        else:
            xai_explain = xai_clean

        s3 = (
            "**3. Cơ sở Suy luận Đồ thị:**\n"
            + xai_explain + "\n\n"
            "_⚠️ Hệ thống chỉ hỗ trợ phân loại lâm sàng — không đưa ra khuyến nghị điều trị._"
        )

        return f"{s1}\n\n{s2}\n\n{s3}"


    def generate_response(
        self,
        context_dict: dict,
        guideline_chunk: str = "",
        use_llm: bool = True,
    ) -> str:
        if not use_llm:
            return self._structured_response(context_dict)

        prompt = f"""Bạn là chuyên gia phân tích lâm sàng. Dựa vào dữ liệu bên dưới, lập báo cáo phân loại mức độ bệnh.
{context_dict['context_str']}

QUY TẮC:
1. Giọng văn y khoa học thuật. KHÔNG XƯNG HÔ.
2. KHÔNG đưa ra khuyến nghị điều trị hay thuốc.
3. Đưa khái niệm y khoa (Cô đặc máu, Giảm tiểu cầu) vào lập luận.
4. Nếu AST/ALT >= 1000 U/L, ưu tiên lập luận SXH Dengue nặng do tổn thương tạng.

Format BẮT BUỘC (3 phần, mỗi phần một đoạn văn liền mạch, không dùng gạch đầu dòng):

**1. Đánh giá Lâm sàng & Cận lâm sàng:**
[Nêu tuổi/nhóm tuổi/ngày bệnh, triệu chứng, diễn tiến xét nghiệm, AST/ALT/Creatinine, các khái niệm nguy cơ]

**2. Giai đoạn & Mức độ:**
[Lập luận phân độ, ưu tiên tổn thương tạng nếu có]

**3. Cơ sở Suy luận Đồ thị (Explainability):**
[Giải thích đường dẫn truy xuất graph, coverage score, lý do chọn phân loại này]

BÁO CÁO:
"""
        if guideline_chunk:
            prompt = prompt.replace(
                "BÁO CÁO:",
                f"HƯỚNG DẪN LÂM SÀNG GỐC TỪ BYT:\n{guideline_chunk}\n\nBÁO CÁO:",
            )
        try:
            response = self._generate_with_llm(prompt)
            clean_text = self._remove_bullets(self._clean_output(response["choices"][0]["text"]))
        except Exception as exc:
            return self._structured_response(context_dict) + f"\n\n_Ghi chú demo: LLM local không sinh được báo cáo ({exc})._"


        # Fallback nếu output không đúng format
        if not ("1." in clean_text and "2." in clean_text and "3." in clean_text):
            return self._structured_response(context_dict)
        # Fallback nếu LLM hallucinate treatment (bất chấp quy tắc)
        _treatment_terms = ["truyền máu", "corticoid", "aspirin", "ibuprofen",
                            "paracetamol", "thuốc", "truyền dịch", "kháng sinh"]
        if any(t in clean_text.lower() for t in _treatment_terms):
            return self._structured_response(context_dict)
        return clean_text

    def generate_response_from_subgraph(
        self,
        patient_id: str,
        subgraph: dict,
        guideline_chunk: str = "",
        method: str = "HGAT",
        use_llm: bool = True,
    ) -> str:
        # ── 1. Full patient demographics + lab values ──────────────────────
        try:
            p_rows = self.db.execute_query("""
                MATCH (p:Patient {id: $pid})
                OPTIONAL MATCH (p)-[:HAS_RECORD]->(dr:DailyRecord)
                WITH p, dr ORDER BY dr.day ASC
                WITH p,
                     collect({day: dr.day, wbc: dr.wbc, plt: dr.plt,
                              hct: dr.hct, hflc: dr.hflc}) AS records,
                     coalesce(p.ast_value,
                              last(collect(dr.ast))) AS ast_value,
                     coalesce(p.alt_value,
                              last(collect(dr.alt))) AS alt_value,
                     coalesce(p.creatinine_value,
                              last(collect(dr.creatinine))) AS creatinine_value
                RETURN p.age AS age, p.gender AS gender,
                       coalesce(p.is_child, false) AS is_child,
                       coalesce(p.current_disease_day, p.admission_day) AS day,
                       ast_value, alt_value, creatinine_value, records,
                       p.plt_nadir AS plt_nadir, p.hct_peak AS hct_peak, p.hflc_peak AS hflc_peak
            """, {"pid": patient_id})
            p = p_rows[0] if p_rows else {}
        except Exception:
            p = {}

        age       = p.get("age", "?")
        is_child  = bool(p.get("is_child", False))
        gender_raw = str(p.get("gender", "")).lower()
        gender    = "Nữ" if gender_raw in ("f", "female", "nữ") else "Nam"
        group     = "nhi" if is_child else "người lớn"
        day       = p.get("day") or "không rõ"
        ast_value = float(p.get("ast_value") or 0)
        alt_value = float(p.get("alt_value") or 0)
        creatinine_value = float(p.get("creatinine_value") or 0)

        demographics_text = (
            f"Tuổi {age}, {gender}, nhóm {group}, ngày bệnh thứ {day}"
        )
        organ_labs_text = (
            f"AST={ast_value if ast_value else 'không có'} U/L, "
            f"ALT={alt_value if alt_value else 'không có'} U/L, "
            f"Creatinine={creatinine_value if creatinine_value else 'không có'} umol/L"
        )

        # Organ damage flags (same logic as retrieve_context)
        organ_flags = []
        if ast_value >= 1000 or alt_value >= 1000:
            organ_flags.append(
                f"AST/ALT ≥ 1000 U/L (thực tế: AST={ast_value:.0f}, ALT={alt_value:.0f}): "
                "tiêu chuẩn SXH Dengue nặng do tổn thương tạng."
            )
        creatinine_ref = 70.0 if is_child else 110.0
        if creatinine_value >= creatinine_ref:
            organ_flags.append(
                f"Creatinine={creatinine_value:.0f} umol/L vượt ngưỡng {creatinine_ref:.0f} "
                f"({'nhi' if is_child else 'người lớn'}): gợi ý tổn thương thận cấp."
            )
        organ_assessment_text = (
            "\n".join(f"- {f}" for f in organ_flags)
            if organ_flags else
            "- Chưa ghi nhận tiêu chí tổn thương tạng rõ từ AST/ALT/Creatinine."
        )

        # Timeline from DailyRecord
        records = p.get("records") or []
        if records and any(r.get("day") is not None for r in records):
            timeline_text = "Diễn tiến cận lâm sàng:\n"
            for r in records:
                if r.get("day") is None:
                    continue
                timeline_text += (
                    f"- Ngày {r['day']}: "
                    f"BC={r.get('wbc') or '-'}, TC={r.get('plt') or '-'}, "
                    f"HCT={r.get('hct') or '-'}, HFLC={r.get('hflc') or '-'}\n"
                )
        else:
            timeline_text = (
                "Chỉ số tổng hợp (EvidenceCase): "
                + subgraph.get("evidence_summary", "Chưa có dữ liệu") + "\n"
            )

        # ── 2. Subgraph data ───────────────────────────────────────────────
        symptoms_text  = ", ".join(subgraph.get("symptoms", [])) or "Không ghi nhận"
        concepts_text  = ", ".join(subgraph.get("concepts", [])) or "Không ghi nhận"
        severity_text  = subgraph.get("severity") or "Chưa xác định"
        matched_rule   = subgraph.get("matched_rule") or "Chưa rõ"
        coverage_score = subgraph.get("coverage_score", 0.0)

        # ── 3. Clinical-friendly XAI (no ML jargon) ───────────────────────
        attention_nodes = subgraph.get("attention_nodes", [])
        clinical_nodes = [
            (name, w) for name, w in attention_nodes
            if not name.startswith("ec_") and name != patient_id
        ]
        if clinical_nodes:
            top_nodes = "; ".join(
                f"{name} ({w:.0%})"
                for name, w in sorted(clinical_nodes, key=lambda x: -x[1])[:4]
            )
            xai_text = (
                f"Hệ thống gán trọng số cao nhất cho các yếu tố: {top_nodes}. "
                f"Phác đồ phù hợp nhất (Coverage {coverage_score:.0%}): {matched_rule}."
            )
        else:
            xai_text = f"Phác đồ tham chiếu: {matched_rule} (Coverage {coverage_score:.0%})."

        plt_nadir = p.get("plt_nadir")
        hct_peak  = p.get("hct_peak")
        hflc_peak = p.get("hflc_peak")
        hematology_summary = (
            f"Tiểu cầu thấp nhất (PLT nadir): {plt_nadir if plt_nadir is not None else 'không có'} G/L; "
            f"Cô đặc máu cao nhất (HCT peak): {hct_peak if hct_peak is not None else 'không có'}%; "
            f"Chỉ số HFLC cao nhất (HFLC peak): {hflc_peak if hflc_peak is not None else 'không có'}%"
        )

        # ── 4. Build context_dict ──────────────────────────────────────────
        context_dict = {
            "symptoms_text":        symptoms_text,
            "timeline_text":        timeline_text,
            "demographics_text":    demographics_text,
            "organ_labs_text":      organ_labs_text,
            "organ_assessment_text": organ_assessment_text,
            "concepts_text":        concepts_text,
            "phase_text":           matched_rule,
            "severity_text":        severity_text,
            "explainability_text":  xai_text,
            "plt_nadir":            plt_nadir,
            "hct_peak":             hct_peak,
            "hflc_peak":            hflc_peak,
            "ast_value":            ast_value,
            "alt_value":            alt_value,
        }
        context_dict["context_str"] = (
            f"Bệnh nhân: {patient_id} — {demographics_text}\n"
            f"Triệu chứng: {symptoms_text}\n"
            f"Chỉ số huyết học tổng hợp: {hematology_summary}\n"
            f"{timeline_text}"
            f"Gan/Thận: {organ_labs_text}\n"
            f"Đánh giá tổn thương tạng:\n{organ_assessment_text}\n"
            f"Khái niệm y khoa: {concepts_text}\n"
            f"Phác đồ: {matched_rule} | Mức độ: {severity_text}\n"
            f"Suy luận: {xai_text}\n"
        )
        return self.generate_response(
            context_dict,
            guideline_chunk=guideline_chunk,
            use_llm=use_llm,
        )


    def ask(self, patient_name: str, use_llm: bool = True) -> str:
        context_data = self.retrieve_context(patient_name)
        if "error" in context_data:
            return context_data["error"]

        guideline = self.retrieve_guideline_chunk(
            context_data.get("rule_name", ""),
            context_data.get("concepts", []),
        )
        return self.generate_response(
            context_data,
            guideline_chunk=guideline,
            use_llm=use_llm,
        )
