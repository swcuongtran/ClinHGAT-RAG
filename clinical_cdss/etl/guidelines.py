import json
import hashlib
import os
from pathlib import Path
from clinical_cdss.core.database import Neo4jConnection
from sentence_transformers import SentenceTransformer

embedder = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_embedder():
    global embedder
    if embedder is None:
        print("⏳ Đang tải mô hình Embedding từ ổ cứng (Offline)...")
        embedder = SentenceTransformer(str(PROJECT_ROOT / "local_models" / "vietnamese-sbert"))
    return embedder

def load_extracted_rules_to_graph(draft_file="data/draft_guidelines.json"):
    if not os.path.exists(draft_file):
        print(f"⚠️ Không tìm thấy file bản nháp tại {draft_file}. Hãy chạy extractor_guidelines.py trước!")
        return

    with open(draft_file, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)

    if not rules_data:
        print("⚠️ File nháp trống rỗng.")
        return

    db = Neo4jConnection()
    emb_model = get_embedder()
    count = 0
    
    # Kiểm tra trùng lặp dựa trên rule_name trước khi nạp
    seen_names = set()
    unique_rules = []
    for rule in rules_data:
        phase_val = rule.get("phase") if rule.get("phase") else "Giai đoạn chung"
        severity_val = rule.get("severity") if rule.get("severity") else "Chưa phân loại"
        key = f"{severity_val}||{phase_val}"
        if key not in seen_names:
            seen_names.add(key)
            unique_rules.append(rule)
    
    removed = len(rules_data) - len(unique_rules)
    if removed > 0:
        print(f"⚠️ Đã loại bỏ {removed} quy tắc bị trùng lặp. Còn lại: {len(unique_rules)} quy tắc.")
    rules_data = unique_rules
    
    print(f"🚀 Bắt đầu nạp {len(rules_data)} quy tắc từ '{draft_file}' vào Neo4j...")
    
    try:
        print("🧹 Đang dọn dẹp các Diagnostic_Rule cũ trong database...")
        db.execute_query("MATCH (r:Diagnostic_Rule) DETACH DELETE r")
        print("✅ Đã dọn dẹp xong Diagnostic_Rule cũ.")
        
        for rule in rules_data:
            phase_val = rule.get("phase") if rule.get("phase") else "Giai đoạn chung"
            severity_val = rule.get("severity") if rule.get("severity") else "Chưa phân loại"
            rule_name = f"{severity_val} ({phase_val})"

            rule_text = " ".join([
                phase_val,
                severity_val,
                rule_name,
                " ".join(rule.get("clinical_signs", []) or []),
                " ".join(rule.get("lab_tests", []) or []),
                " ".join(rule.get("treatments", []) or []),
                " ".join(rule.get("contraindications", []) or []),
            ])
            rule_embedding = emb_model.encode(rule_text).tolist()

            query_core = """
            MERGE (sev:Severity {name: $severity})
            MERGE (rule:Diagnostic_Rule {name: $rule_name})
            SET rule.phase = $phase,
                rule.embedding = $rule_embedding
            MERGE (rule)-[:DETERMINES]->(sev)
            """
            db.execute_query(query_core, parameters={
                "severity": severity_val,
                "rule_name": rule_name,
                "phase": phase_val,
                "rule_embedding": rule_embedding,
            })

            # Tạo chuỗi text phục vụ cho ánh xạ Concept cứng: Chỉ lấy từ thông tin chẩn đoán
            # (Giai đoạn, Phân độ, Tên quy tắc, Triệu chứng lâm sàng, Xét nghiệm cận lâm sàng)
            # loại trừ treatments và contraindications để tránh nhận nhầm triệu chứng đầu vào.
            rule_concept_text = " ".join([
                phase_val,
                severity_val,
                rule_name,
                " ".join(rule.get("clinical_signs", []) or []),
                " ".join(rule.get("lab_tests", []) or []),
            ]).lower()

            RULE_TEXT_CONCEPT_MAP = {
                "giai đoạn sốt": "Giai đoạn sốt",
                "giai đoạn nguy hiểm": "Giai đoạn nguy hiểm theo ngày bệnh",
                "giai đoạn hồi phục": "Giai đoạn hồi phục",
                "sốt cao": "Giai đoạn sốt",
                "trẻ": "Bệnh nhi",
                "bệnh nhi": "Bệnh nhi",
                "người lớn": "Người lớn",
                "dấu hiệu cảnh báo": "Có dấu hiệu cảnh báo",
                "nhiều dấu hiệu cảnh báo": "Nhiều dấu hiệu cảnh báo",
                "đau bụng": "Đau bụng cảnh báo",
                "nôn": "Nôn ói",
                "tiêu chảy": "Tiêu chảy",
                "gan to": "Gan to",
                "lừ đừ": "Lừ đừ / Li bì",
                "li bì": "Lừ đừ / Li bì",
                "sốc": "Sốc Dengue",
                "tái sốc": "Tái sốc",
                "sốc nặng": "Sốc Dengue nặng",
                "sốc dengue nặng": "Sốc Dengue nặng",
                "sốc sxhd nặng": "Sốc Dengue nặng",
                "hct tăng": "Cô đặc máu",
                "hematocrit": "Cô đặc máu",
                "tiểu cầu": "Giảm tiểu cầu",
                "tiểu cầu giảm": "Giảm tiểu cầu",
                "giảm tiểu cầu": "Giảm tiểu cầu",
                "hct tăng > 20": "HCT tăng >=20%",
                "hct tăng ≥ 20": "HCT tăng >=20%",
                "bạch cầu": "Bạch cầu giảm dần",
                "xuất huyết": "Xuất huyết",
                "xuất huyết nặng": "Xuất huyết nặng",
                "xuất huyết niêm mạc": "Xuất huyết niêm mạc",
                "chảy máu mũi": "Chảy máu mũi",
                "chảy máu chân răng": "Chảy máu chân răng",
                "xuất huyết âm đạo": "Xuất huyết âm đạo",
                "nôn ra máu": "Xuất huyết tiêu hóa",
                "tiêu phân đen": "Xuất huyết tiêu hóa",
                "xuất huyết tiêu hóa": "Xuất huyết tiêu hóa",
                "mảng bầm": "Xuất huyết mô mềm / bầm máu",
                "bầm tím": "Xuất huyết mô mềm / bầm máu",
                "chấm xuất huyết": "Chấm xuất huyết",
                "vàng da": "Vàng da",
                "men gan": "Tăng men gan",
                "gan": "Tổn thương gan nặng / suy gan cấp",
                "suy gan": "Tổn thương gan nặng / suy gan cấp",
                "tổn thương tạng": "Tổn thương tạng",
                "suy tạng": "Tổn thương tạng",
                "suy các tạng": "Tổn thương tạng",
                "thận": "Tổn thương thận cấp / giảm tưới máu thận",
                "cơ tim": "Tổn thương cơ tim",
                "suy tim": "Tổn thương cơ tim",
                "hô hấp": "Suy hô hấp",
                "thoát huyết tương": "Giảm albumin / thoát huyết tương",
                "đông máu": "Rối loạn đông máu",
                "dấu hiệu cảnh báo": "Có dấu hiệu cảnh báo",
            }
            for keyword, concept_name in RULE_TEXT_CONCEPT_MAP.items():
                if keyword in rule_concept_text:
                    db.execute_query("""
                        MATCH (rule:Diagnostic_Rule {name: $rule_name})
                        MERGE (c:Concept {name: $concept_name})
                        MERGE (c)-[:LINKED_TO_RULE]->(rule)
                    """, {"rule_name": rule_name, "concept_name": concept_name})

            valid_signs = [s for s in rule.get("clinical_signs", []) if s and s.lower() not in ["chưa có", "không có", ""]]
            if valid_signs:
                symptoms_data = [{"name": s, "embedding": emb_model.encode(s).tolist()} for s in valid_signs]
                q_sign = """
                MATCH (rule:Diagnostic_Rule {name: $rule_name})
                UNWIND $symptoms_data AS sym_data
                MERGE (sym:Symptom {name: sym_data.name})
                ON CREATE SET sym.embedding = sym_data.embedding
                ON MATCH SET sym.embedding = sym_data.embedding
                MERGE (sym)-[:PART_OF_RULE]->(rule)
                """
                db.execute_query(q_sign, {"rule_name": rule_name, "symptoms_data": symptoms_data})
                
                # Tạo cầu nối ngữ nghĩa Symptom -> Concept (Semantic Bridge)
                for sym_name in valid_signs:
                    sym_lower = sym_name.lower()
                    for keyword, concept_name in RULE_TEXT_CONCEPT_MAP.items():
                        if keyword in sym_lower:
                            q_sym_bridge = """
                            MERGE (s:Symptom {name: $sym_name})
                            MERGE (c:Concept {name: $concept_name})
                            MERGE (s)-[:MAPS_TO]->(c)
                            """
                            db.execute_query(q_sym_bridge, {"sym_name": sym_name, "concept_name": concept_name})

            valid_labs = [l for l in rule.get("lab_tests", []) if l and l.lower() not in ["chưa có", "không có", ""]]
            if valid_labs:
                q_lab = """
                MATCH (rule:Diagnostic_Rule {name: $rule_name})
                UNWIND $labs AS lab
                MERGE (l:LabTest {name: lab})
                MERGE (l)-[:PART_OF_RULE]->(rule)
                """
                db.execute_query(q_lab, {"rule_name": rule_name, "labs": valid_labs})
                
                # Tạo cầu nối ngữ nghĩa LabTest -> Concept (Semantic Bridge)
                # Dùng bảng ánh xạ từ số liệu xét nghiệm sang khái niệm y khoa
                LABTEST_CONCEPT_MAP = {
                    "hct": "Cô đặc máu",
                    "hematocrit": "Cô đặc máu",
                    "tiểu cầu": "Giảm tiểu cầu",
                    "platelet": "Giảm tiểu cầu",
                    "plt": "Giảm tiểu cầu",
                    "tiểu cầu giảm": "Giảm tiểu cầu",
                    "giảm tiểu cầu": "Giảm tiểu cầu",
                    "bạch cầu": "Hạ bạch cầu",
                    "wbc": "Hạ bạch cầu",
                    "bạch cầu giảm": "Bạch cầu giảm dần",
                    "sốc": "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)",
                    "huyết áp": "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)",
                    "mạch": "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)",
                    "albumin": "Giảm albumin / thoát huyết tương",
                    "thoát huyết tương": "Giảm albumin / thoát huyết tương",
                    "dịch màng": "Giảm albumin / thoát huyết tương",
                    "tràn dịch": "Giảm albumin / thoát huyết tương",
                    "đông máu": "Rối loạn đông máu",
                    "tq": "Rối loạn đông máu",
                    "tck": "TCK kéo dài",
                    "fibrinogen": "Giảm fibrinogen",
                    "bilirubin": "Tăng bilirubin",
                    "troponin": "Tổn thương cơ tim",
                    "cơ tim": "Tổn thương cơ tim",
                    "suy tim": "Tổn thương cơ tim",
                    "suy hô hấp": "Suy hô hấp",
                    "ast": "Tổn thương gan nặng / suy gan cấp",
                    "alt": "Tổn thương gan nặng / suy gan cấp",
                    "gan": "Tổn thương gan nặng / suy gan cấp",
                    "creatinine": "Tổn thương thận cấp / giảm tưới máu thận",
                    "creatinin": "Tổn thương thận cấp / giảm tưới máu thận",
                    "thận": "Tổn thương thận cấp / giảm tưới máu thận",
                }
                for lab_name in valid_labs:
                    lab_lower = lab_name.lower()
                    for keyword, concept_name in LABTEST_CONCEPT_MAP.items():
                        if keyword in lab_lower:
                            q_bridge = """
                            MERGE (l:LabTest {name: $lab_name})
                            MERGE (c:Concept {name: $concept_name})
                            MERGE (l)-[:MAPS_TO]->(c)
                            """
                            db.execute_query(q_bridge, {"lab_name": lab_name, "concept_name": concept_name})

                            # Nối trực tiếp Concept → Rule (hoàn thiện hyperedge cho HGAT)
                            q_concept_rule = """
                            MATCH (rule:Diagnostic_Rule {name: $rule_name})
                            MERGE (c:Concept {name: $concept_name})
                            MERGE (c)-[:LINKED_TO_RULE]->(rule)
                            """
                            db.execute_query(q_concept_rule, {
                                "rule_name": rule_name,
                                "concept_name": concept_name
                            })
                            break


            valid_treats = [t for t in rule.get("treatments", []) if t and t.lower() not in ["chưa có", "không có", ""]]
            if valid_treats:
                q_treat = """
                MATCH (rule:Diagnostic_Rule {name: $rule_name})
                UNWIND $treats AS treat
                MERGE (t:Treatment {action: treat})
                MERGE (rule)-[:RECOMMENDS]->(t)
                """
                db.execute_query(q_treat, {"rule_name": rule_name, "treats": valid_treats})

            valid_contras = [c for c in rule.get("contraindications", []) if c and c.lower() not in ["chưa có", "không có", ""]]
            if valid_contras:
                q_contra = """
                MATCH (rule:Diagnostic_Rule {name: $rule_name})
                UNWIND $contras AS contra
                MERGE (c:Contraindication {action: contra})
                MERGE (rule)-[:AVOIDS]->(c)
                """
                db.execute_query(q_contra, {"rule_name": rule_name, "contras": valid_contras})

            count += 1
    finally:
        db.close()
        print(f"✅ Đã lưu {count} Cụm bệnh cảnh vào Siêu đồ thị (Hypergraph).")

def _chunk_text(text: str, max_chars: int = 1800, overlap: int = 250):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def index_guideline_chunks(
    pdf_path="data/Huong-dan-chan-doan-va-dieu-tri.pdf",
    start_page=207,
    end_page=278,
    chunk_chars=1800,
    overlap=250,
):
    """Store source guideline text chunks in Neo4j for grounded RAG retrieval."""
    if not os.path.exists(pdf_path):
        print(f"Guideline PDF not found: {pdf_path}")
        return 0

    import fitz

    emb_model = get_embedder()
    db = Neo4jConnection()
    rows = []
    try:
        doc = fitz.open(pdf_path)
        first_page = max(0, start_page - 1)
        last_page = min(len(doc) - 1, end_page - 1)
        for page_idx in range(first_page, last_page + 1):
            page_no = page_idx + 1
            page_text = doc[page_idx].get_text("text")
            for chunk_idx, chunk in enumerate(_chunk_text(page_text, chunk_chars, overlap)):
                raw_id = f"{pdf_path}:{page_no}:{chunk_idx}:{chunk[:80]}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
                rows.append({
                    "id": chunk_id,
                    "text": chunk,
                    "source": os.path.basename(pdf_path),
                    "page_start": page_no,
                    "page_end": page_no,
                    "chunk_index": chunk_idx,
                    "embedding": emb_model.encode(chunk).tolist(),
                })
        doc.close()

        if rows:
            db.execute_query("""
                UNWIND $rows AS row
                MERGE (g:GuidelineChunk {id: row.id})
                SET g.text = row.text,
                    g.source = row.source,
                    g.page_start = row.page_start,
                    g.page_end = row.page_end,
                    g.chunk_index = row.chunk_index,
                    g.embedding = row.embedding
            """, {"rows": rows})
        print(f"Indexed {len(rows)} guideline chunks.")
        return len(rows)
    finally:
        db.close()


if __name__ == "__main__":
    load_extracted_rules_to_graph()
    index_guideline_chunks()
