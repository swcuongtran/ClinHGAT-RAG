import pandas as pd
import numpy as np
import sys
from pathlib import Path
from clinical_cdss.core.database import Neo4jConnection
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_slope(day_value_pairs: list) -> float:
    """Tính slope tuyến tính đơn giản cho WBC trend."""
    if len(day_value_pairs) < 2:
        return 0.0
    days = [d for d, v in day_value_pairs]
    vals = [v for d, v in day_value_pairs]
    n = len(days)
    mean_d = sum(days) / n
    mean_v = sum(vals) / n
    num = sum((d - mean_d) * (v - mean_v) for d, v in zip(days, vals))
    den = sum((d - mean_d) ** 2 for d in days)
    return num / den if den != 0 else 0.0


def _num(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _text(value) -> str:
    return "" if pd.isna(value) else str(value).strip().lower()


def infer_patient_concepts(
    row,
    *,
    is_child: bool,
    is_male: bool,
    age: float,
    admission_day: int,
    plt_nadir,
    hct_peak,
    hct_change_pct: float,
    hflc_peak,
    wbc_trend: float,
    ast_value: float,
    alt_value: float,
    creatinine_value: float,
    critical_day,
):
    """Infer clinical concepts available from the CSV columns."""
    concepts = []

    def add(enabled, name):
        if enabled and name not in concepts:
            concepts.append(name)

    hct_threshold = 38.0 if is_child else (45.0 if is_male else 40.0)
    note = _text(row.get("ghichutinhtrangbenhnang"))
    bleed_location = _text(row.get("Vitrixuathuyet"))
    other_symptoms = _text(row.get("Trieuchungkhac"))
    comorbidity = _text(row.get("benhlynen"))

    has_low_plt = plt_nadir is not None and plt_nadir < 100
    has_high_hct = hct_peak is not None and hct_peak > hct_threshold

    # ── Nhân khẩu & nhóm bệnh nhân ──────────────────────────────────────
    add(is_child, "Bệnh nhi")
    add(not is_child, "Người lớn")
    add(age <= 1, "Nhũ nhi")
    add(_text(row.get("pregnancy")) not in {"", "nan"}, "Thai kỳ")

    # ── Giai đoạn bệnh (BYT Phụ lục 1) ──────────────────────────────────
    # BYT: Sốt ngày 1-3 / Nguy hiểm ngày 4-6 / Hồi phục ngày 7+
    add(admission_day <= 3, "Giai đoạn sốt")
    add(4 <= admission_day <= 6, "Giai đoạn nguy hiểm theo ngày bệnh")
    add(admission_day >= 7, "Giai đoạn hồi phục")

    # ── Triệu chứng lâm sàng ─────────────────────────────────────────────
    add(_num(row.get("NVnonoi")) == 1, "Nôn ói")
    add(_num(row.get("NVtieuchay")) == 1, "Tiêu chảy")
    add(_num(row.get("NVdaubung")) == 1, "Đau bụng cảnh báo")
    add(_num(row.get("NVganto")) == 1, "Gan to")
    add(_num(row.get("NVxuathuyet")) == 1, "Xuất huyết")
    add("vangda" in other_symptoms or "vàng" in other_symptoms, "Vàng da")

    # Vị trí xuất huyết
    add(_num(row.get("Petechia")) == 1 or "petech" in bleed_location, "Chấm xuất huyết")
    add(any(k in bleed_location for k in ["am dao", "amdao", "rong kinh"]), "Xuất huyết âm đạo")
    add(any(k in bleed_location for k in ["mui", "cam"]), "Chảy máu mũi")
    add(any(k in bleed_location for k in ["rang", "chanrang", "chaymaurang"]), "Chảy máu chân răng")
    add(any(k in bleed_location or k in other_symptoms
            for k in ["nonramau", "tieuphanden", "phan den", "tiêu phân đen"]),
        "Xuất huyết tiêu hóa")
    add(any(k in bleed_location for k in ["hematoma", "bammau", "bầm"]),
        "Xuất huyết mô mềm / bầm máu")
    # BYT: xuất huyết niêm mạc (mũi, răng, âm đạo) = dấu hiệu cảnh báo riêng
    _mucosa_bleed = any(k in bleed_location for k in
                        ["mui", "cam", "rang", "chanrang", "am dao", "amdao"])
    add(_mucosa_bleed, "Xuất huyết niêm mạc")

    # ── 6 Dấu hiệu cảnh báo BYT (Phụ lục 2) ─────────────────────────────
    _lethargy = any(k in note for k in ["liduli", "ludu", "lừ đừ", "li bì", "bứt rứt"]) \
             or any(k in other_symptoms for k in ["liduli", "ludu", "lừ đừ", "li bì", "bứt rứt"])
    add(_lethargy, "Lừ đừ / Li bì")

    warning_count = sum([
        _num(row.get("NVdaubung")) == 1,       # 1. Đau bụng
        _num(row.get("NVnonoi")) == 1,          # 2. Nôn ói liên tục
        _mucosa_bleed,                          # 3. Xuất huyết niêm mạc
        _num(row.get("NVganto")) == 1,          # 4. Gan to > 2cm
        _lethargy,                              # 5. Lừ đừ / li bì / bứt rứt
        has_high_hct and has_low_plt,           # 6. HCT tăng đồng thời PLT giảm nhanh
    ])
    add(warning_count >= 1, "Có dấu hiệu cảnh báo")
    add(warning_count >= 3, "Nhiều dấu hiệu cảnh báo")

    # ── Huyết học ─────────────────────────────────────────────────────────
    add(has_low_plt, "Giảm tiểu cầu")
    add(plt_nadir is not None and plt_nadir < 50, "Giảm tiểu cầu nặng")
    add(plt_nadir is not None and plt_nadir < 20, "Giảm tiểu cầu rất nặng")
    add(plt_nadir is not None and plt_nadir < 10, "Giảm tiểu cầu nguy kịch")
    add(has_high_hct, "Cô đặc máu")
    add(hct_change_pct >= 20, "HCT tăng >=20%")          # BYT: ≥ 20% so với ban đầu
    add(has_low_plt and has_high_hct, "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)")
    add(critical_day is not None and 4 <= int(critical_day) <= 6,
        "Nadir tiểu cầu trong pha nguy hiểm")
    add(wbc_trend < -0.5, "Bạch cầu giảm dần")
    add(wbc_trend > 1.0, "Bạch cầu tăng nhanh")
    # HFLC: chỉ số nghiên cứu, không có trong BYT 2019
    add(hflc_peak is not None and hflc_peak >= 1.0, "HFLC tăng")
    add(hflc_peak is not None and hflc_peak >= 2.0, "HFLC tăng cao")

    # ── Sinh hoá / Tổn thương tạng (BYT: tiêu chuẩn SXHD nặng) ──────────
    add(ast_value >= 200 or alt_value >= 200, "Tăng men gan")
    add(ast_value >= 1000 or alt_value >= 1000, "Tổn thương gan nặng / suy gan cấp")
    add(creatinine_value >= (70.0 if is_child else 110.0),
        "Tổn thương thận cấp / giảm tưới máu thận")
    add(_num(row.get("Albumin")) > 0 and _num(row.get("Albumin")) < 35,
        "Giảm albumin / thoát huyết tương")
    add(_num(row.get("TQphantram")) > 0 and _num(row.get("TQphantram")) < 70,
        "Rối loạn đông máu")
    add(_num(row.get("Fibrinogen")) > 0 and _num(row.get("Fibrinogen")) < 2,
        "Giảm fibrinogen")
    add(_num(row.get("TCKgiay")) > 0 and _num(row.get("TCKgiay")) > 45, "TCK kéo dài")
    add(_num(row.get("Bilirubin")) > 0 and _num(row.get("Bilirubin")) > 20, "Tăng bilirubin")
    add(_num(row.get("Troponin")) > 0, "Tổn thương cơ tim")

    add("soc" in note or "sốc" in note, "Sốc Dengue")
    add("soctaisoc" in note or "tái sốc" in note, "Tái sốc")
    add("socsxhnang" in note or "sốc nặng" in note, "Sốc Dengue nặng")
    add("xhth" in note or "xuathuyet" in note or "xuất huyết" in note, "Xuất huyết nặng")
    # Fix: "gan" quá rộng — dùng keywords cụ thể tránh match "bệnh gan nền"
    add(
        any(k in note for k in ["suigan", "suy gan", "tonthuongan", "tổn thương gan"]),
        "Tổn thương gan nặng / suy gan cấp"
    )
    # Fix: "than" quá rộng (có thể match "than thở") — dùng keywords cụ thể
    add(
        any(k in note for k in ["tonthuongtang", "tổn thương tạng", "suitan", "suy tạng"]),
        "Tổn thương tạng"
    )
    add("shh" in note, "Suy hô hấp")

    add(bool(comorbidity), "Có bệnh nền")
    add("tanghuyetap" in comorbidity or "tha" in comorbidity, "Tăng huyết áp nền")
    add("dtd" in comorbidity or "daithaoduong" in comorbidity, "Đái tháo đường nền")
    add("hen" in comorbidity, "Hen phế quản nền")
    add("hbv" in comorbidity or "gan" in comorbidity, "Bệnh gan nền")
    add("tim" in comorbidity or "thonglienthat" in comorbidity, "Bệnh tim nền")

    return concepts


def refresh_evidence_rule_matches(
    db: Neo4jConnection,
    threshold: float = 0.15,
    evidence_id: str | None = None,
):
    """Materialize EvidenceCase -> Diagnostic_Rule matches for HGAT and XAI traversal.

    When ``evidence_id`` is provided, only that EvidenceCase is refreshed. This keeps
    online daily updates responsive instead of rebuilding MATCHES for the full graph.
    """
    db.execute_query("""
        MATCH (ec:EvidenceCase)-[m:MATCHES]->(:Diagnostic_Rule)
        WHERE $evidence_id IS NULL OR ec.id = $evidence_id
        DELETE m
    """, {"evidence_id": evidence_id})
    result = db.execute_query("""
        MATCH (ec:EvidenceCase)
        WHERE $evidence_id IS NULL OR ec.id = $evidence_id
        OPTIONAL MATCH (ec)-[:HAS_SYMPTOM]->(ec_sym:Symptom)
        WITH ec, collect(DISTINCT ec_sym) AS ec_symptoms
        OPTIONAL MATCH (ec)-[:HAS_CONCEPT]->(ec_con:Concept)
        WITH ec, ec_symptoms, collect(DISTINCT ec_con) AS ec_concepts
        MATCH (rule:Diagnostic_Rule)
        OPTIONAL MATCH (rule)<-[:PART_OF_RULE]-(rule_sym:Symptom)
        WITH ec, ec_symptoms, ec_concepts, rule, collect(DISTINCT rule_sym) AS rule_symptoms
        OPTIONAL MATCH (rule)<-[:LINKED_TO_RULE]-(rule_con:Concept)
        WITH ec, ec_symptoms, ec_concepts, rule, rule_symptoms,
             collect(DISTINCT rule_con) AS rule_concepts
        WHERE ALL(c IN rule_concepts WHERE NOT c.name IN ['Bệnh nhi', 'Người lớn', 'Nhũ nhi', 'Thai kỳ'] OR c IN ec_concepts)
        
        // Xác định các Concept khớp giữa rule và bệnh nhân
        WITH ec, ec_symptoms, ec_concepts, rule, rule_symptoms, rule_concepts,
             [c IN rule_concepts WHERE c IN ec_concepts] AS matched_concepts
             
        // Tính số lượng Concept lâm sàng khớp (loại trừ các concept ngữ cảnh/phi lâm sàng)
        WITH ec, ec_symptoms, ec_concepts, rule, rule_symptoms, rule_concepts, matched_concepts,
             size([c IN matched_concepts WHERE NOT c.name IN [
                 'Giai đoạn nguy hiểm theo ngày bệnh', 'Giai đoạn sốt', 'Giai đoạn hồi phục', 
                 'Giai đoạn sớm', 'Giai đoạn chuyển tiếp nguy hiểm/hồi phục', 
                 'Bệnh nhi', 'Người lớn', 'Nhũ nhi', 'Thai kỳ'
             ]]) AS clinical_concept_match,
             size(matched_concepts) AS concept_match,
             size(rule_concepts) AS concept_total
             
        // Tính số lượng triệu chứng khớp (bao gồm so khớp mềm qua Concept chung)
        WITH ec, rule, concept_match, concept_total, clinical_concept_match,
             size([rs IN rule_symptoms WHERE rs IN ec_symptoms OR EXISTS {
                 MATCH (rs)-[:MAPS_TO]->(c:Concept)<-[:MAPS_TO]-(ps:Symptom)
                 WHERE ps IN ec_symptoms
             }]) AS sym_match,
             size(rule_symptoms) AS sym_total
             
        WITH ec, rule, sym_match, sym_total, concept_match, concept_total, clinical_concept_match,
             CASE WHEN sym_total > 0
                  THEN toFloat(sym_match) / toFloat(sym_total)
                  ELSE 0.0 END AS sym_score,
             CASE WHEN concept_total > 0
                  THEN toFloat(concept_match) / toFloat(concept_total)
                  ELSE 0.0 END AS concept_score
                  
        WITH ec, rule, sym_match, sym_total, concept_match, concept_total, clinical_concept_match,
             (0.6 * sym_score) + (0.4 * concept_score) AS coverage_score
             
        WHERE coverage_score >= $threshold
          // Bộ lọc: loại bỏ luật rỗng hoặc chỉ khớp thông tin ngữ cảnh thời gian/tuổi
          AND NOT (sym_match = 0 AND clinical_concept_match = 0)
          
        MERGE (ec)-[m:MATCHES]->(rule)
        SET m.coverage_score = coverage_score,
            m.sym_match = sym_match,
            m.sym_total = sym_total,
            m.concept_match = concept_match,
            m.concept_total = concept_total
        RETURN count(m) AS n
    """, {"threshold": threshold, "evidence_id": evidence_id})
    return result[0]["n"] if result else 0


def load_patients_to_graph(file_path: str):
    db = Neo4jConnection()
    try:
        print("🧹 Đang dọn dẹp dữ liệu bệnh nhân cũ trong database...")
        db.execute_query("MATCH (p:Patient) DETACH DELETE p")
        db.execute_query("MATCH (ec:EvidenceCase) DETACH DELETE ec")
        db.execute_query("MATCH (dr:DailyRecord) DETACH DELETE dr")
        print("✅ Đã dọn dẹp xong dữ liệu bệnh nhân cũ.")

        df = pd.read_csv(file_path).drop(index=0)
        # Loại bỏ trùng lặp tên bệnh nhân, giữ lại bản ghi đầu tiên
        initial_len = len(df)
        df = df.drop_duplicates(subset=['Ten'], keep='first')
        print(f"📋 Đã lọc trùng lặp bệnh nhân: {initial_len} -> {len(df)} hàng dữ liệu duy nhất.")

        print("⏳ Đang tải Embedding Model (Offline)...")
        embedder = SentenceTransformer(str(PROJECT_ROOT / "local_models" / "vietnamese-sbert"))

        symptoms_map = {
            'NVnonoi': 'Nôn ói', 'NVtieuchay': 'Tiêu chảy', 'NVdaubung': 'Đau bụng',
            'NVdauco': 'Đau cơ', 'NVdaudau': 'Đau đầu', 'NVdausauhocmat': 'Đau sau hốc mắt',
            'NVho': 'Ho', 'NVxuathuyet': 'Xuất huyết', 'NVganto': 'Gan to',
            'Petechia': 'Chấm xuất huyết (Petechiae)'
        }

        # Vector search 1 lần cho tất cả triệu chứng
        print("🔍 Đang quét Vector Search cho triệu chứng chuẩn...")
        canonical_symptoms = {}
        for raw_symptom in symptoms_map.values():
            vector = embedder.encode(raw_symptom).tolist()
            result = db.execute_query("""
                CALL db.index.vector.queryNodes('symptom_embedding_index', 1, $vector)
                YIELD node AS canonical_sym, score
                WHERE score > 0.85
                RETURN canonical_sym.name AS name
            """, parameters={'vector': vector})
            canonical_symptoms[raw_symptom] = result[0]['name'] if result else raw_symptom

        count = 0
        patient_symptoms_map = {}  # lưu để tạo SIMILAR_TO sau

        print("🧑‍⚕️ Đang nạp dữ liệu bệnh nhân...")
        for _, row in df.iterrows():
            patient_name = str(row['Ten']).strip()
            if pd.isna(row['Ten']) or not patient_name:
                continue

            row_lower = {str(k).lower(): v for k, v in row.items()}

            # ── Nhân khẩu học ─────────────────────────────────────────────
            gender_raw = str(row['gioitinh']).strip().lower()
            is_male  = gender_raw in ['1', '1.0', 'nam', 'm', 'male']
            try:
                age = float(row['Tuoi'])
            except Exception:
                age = 18.0
            is_child = age <= 15

            try:
                admission_day = int(float(row.get('ngaybenhlucnhapvien', 4)))
            except Exception:
                admission_day = 4

            # ── Nhãn chẩn đoán ────────────────────────────────────────────
            label_raw = row.get('benhcanhcuoicunglagi')
            label_3class  = int(float(label_raw)) if pd.notna(label_raw) else None
            binary_label  = (1 if label_3class == 3 else 0) if label_3class is not None else None

            # ── Patient node ──────────────────────────────────────────────
            db.execute_query("""
                MERGE (p:Patient {id: $name})
                SET p.age             = $age,
                    p.gender          = $gender,
                    p.weight          = $weight,
                    p.is_child        = $is_child,
                    p.admission_day   = $admission_day,
                    p.diagnosis_label = $label_3class,
                    p.binary_label    = $binary_label
            """, parameters={
                'name': patient_name, 'age': age,
                'gender': 1 if is_male else 2,
                'weight': float(row['Cannang']) if pd.notna(row.get('Cannang')) else None,
                'is_child': is_child, 'admission_day': admission_day,
                'label_3class': label_3class, 'binary_label': binary_label
            })

            # ── Triệu chứng ───────────────────────────────────────────────
            patient_sym_list = []
            for col_name, raw_symptom in symptoms_map.items():
                if pd.notna(row.get(col_name)) and row[col_name] == 1.0:
                    cname = canonical_symptoms[raw_symptom]
                    db.execute_query("""
                        MATCH (p:Patient {id: $pname})
                        MERGE (s:Symptom {name: $cname})
                        MERGE (p)-[:HAS_SYMPTOM]->(s)
                    """, parameters={'pname': patient_name, 'cname': cname})
                    patient_sym_list.append(cname)

            patient_symptoms_map[patient_name] = patient_sym_list

            if pd.notna(row.get('Vitrixuathuyet')):
                db.execute_query("""
                    MATCH (p:Patient {id: $name})-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE s.name CONTAINS 'Xuất huyết'
                    SET r.location = $loc
                """, parameters={'name': patient_name,
                                  'loc': str(row['Vitrixuathuyet']).strip()})

            # ── Dữ liệu theo ngày (Disease-Day Aligned) ───────────────────
            daily_records = []
            for hospital_day in range(1, 11):
                wbc  = row_lower.get(f'bachcaun{hospital_day}')
                plt_ = row_lower.get(f'tieucaun{hospital_day}')
                hct  = row_lower.get(f'hctn{hospital_day}')
                hflc = row_lower.get(f'hflcn{hospital_day}')
                if any(pd.notna(x) for x in [wbc, plt_, hct, hflc]):
                    daily_records.append({
                        "day":         hospital_day,
                        "disease_day": admission_day + hospital_day - 1,
                        "wbc":  float(wbc)  if pd.notna(wbc)  else None,
                        "plt":  float(plt_) if pd.notna(plt_) else None,
                        "hct":  float(hct)  if pd.notna(hct)  else None,
                        "hflc": float(hflc) if pd.notna(hflc) else None,
                    })

            if daily_records:
                db.execute_query("""
                    MATCH (p:Patient {id: $pname})
                    UNWIND $records AS rec
                    MERGE (dr:DailyRecord {id: p.id + '_day_' + toString(rec.day)})
                    SET dr.day = rec.day, dr.disease_day = rec.disease_day,
                        dr.wbc = rec.wbc, dr.plt = rec.plt,
                        dr.hct = rec.hct, dr.hflc = rec.hflc
                    MERGE (p)-[:HAS_RECORD]->(dr)
                """, parameters={"pname": patient_name, "records": daily_records})

            # ── Aggregate Features ─────────────────────────────────────────
            plt_vals  = [r['plt']  for r in daily_records if r['plt']  is not None]
            hct_vals  = [r['hct']  for r in daily_records if r['hct']  is not None and r['hct'] <= 100]
            hflc_vals = [r['hflc'] for r in daily_records if r['hflc'] is not None]
            wbc_pairs = [(r['disease_day'], r['wbc']) for r in daily_records if r['wbc'] is not None]

            plt_nadir     = min(plt_vals)  if plt_vals  else None
            plt_below10   = any(p < 10 for p in plt_vals)
            hct_peak      = max(hct_vals)  if hct_vals  else None
            hflc_peak     = max(hflc_vals) if hflc_vals else None
            wbc_trend     = compute_slope(wbc_pairs)
            hct_baseline  = hct_vals[0] if hct_vals else None
            hct_change_pct = (
                (hct_peak - hct_baseline) / hct_baseline * 100
                if hct_peak and hct_baseline and hct_baseline > 0 else 0.0
            )
            ast_raw       = row.get('AST')
            alt_raw       = row.get('ALT')
            creat_raw     = row.get('Creatinin')
            ast_value     = min(float(ast_raw), 5000.0) if pd.notna(ast_raw) else 0.0
            alt_value     = float(alt_raw) if pd.notna(alt_raw) else 0.0
            creatinine_value = float(creat_raw) if pd.notna(creat_raw) else 0.0
            ast_available = 1 if pd.notna(ast_raw) else 0
            plt_with_day  = [(r['disease_day'], r['plt']) for r in daily_records if r['plt'] is not None]
            critical_day  = min(plt_with_day, key=lambda x: x[1])[0] if plt_with_day else admission_day

            # ── Concept Mapping ────────────────────────────────────────────
            concept_names = infer_patient_concepts(
                row,
                is_child=is_child,
                is_male=is_male,
                age=age,
                admission_day=admission_day,
                plt_nadir=plt_nadir,
                hct_peak=hct_peak,
                hct_change_pct=hct_change_pct,
                hflc_peak=hflc_peak,
                wbc_trend=wbc_trend,
                ast_value=ast_value,
                alt_value=alt_value,
                creatinine_value=creatinine_value,
                critical_day=critical_day,
            )

            db.execute_query("""
                MATCH (p:Patient {id: $name})
                OPTIONAL MATCH (p)-[old:HAS_CONDITION]->(:Concept)
                DELETE old
            """, parameters={'name': patient_name})

            for concept_name in concept_names:
                db.execute_query("""
                    MATCH (p:Patient {id: $name})
                    MERGE (c:Concept {name: $cname})
                    MERGE (p)-[:HAS_CONDITION]->(c)
                """, parameters={'name': patient_name, 'cname': concept_name})

            # ── Cập nhật aggregate features lên Patient node ──────────────
            db.execute_query("""
                MATCH (p:Patient {id: $name})
                SET p.plt_nadir      = $plt_nadir,
                    p.plt_below10    = $plt_below10,
                    p.hct_peak       = $hct_peak,
                    p.hct_change_pct = $hct_change_pct,
                    p.hflc_peak      = $hflc_peak,
                    p.wbc_trend      = $wbc_trend,
                    p.ast_value      = $ast_value,
                    p.alt_value      = $alt_value,
                    p.creatinine_value = $creatinine_value,
                    p.ast_available  = $ast_available
            """, parameters={
                'name': patient_name,
                'plt_nadir': plt_nadir,    'plt_below10': plt_below10,
                'hct_peak': hct_peak,      'hct_change_pct': hct_change_pct,
                'hflc_peak': hflc_peak,    'wbc_trend': wbc_trend,
                'ast_value': ast_value,    'alt_value': alt_value,
                'creatinine_value': creatinine_value,
                'ast_available': ast_available
            })

            # ── EvidenceCase node ─────────────────────────────────────────
            ec_id = f"ec_{patient_name}"

            db.execute_query("""
                MERGE (ec:EvidenceCase {id: $ec_id})
                SET ec.plt_nadir      = $plt_nadir,
                    ec.plt_below10    = $plt_below10,
                    ec.hct_peak       = $hct_peak,
                    ec.hct_change_pct = $hct_change_pct,
                    ec.hflc_peak      = $hflc_peak,
                    ec.wbc_trend      = $wbc_trend,
                    ec.ast_value      = $ast_value,
                    ec.alt_value      = $alt_value,
                    ec.creatinine_value = $creatinine_value,
                    ec.ast_available  = $ast_available,
                    ec.critical_day   = $critical_day
                WITH ec
                MATCH (p:Patient {id: $pname})
                MERGE (p)-[:HAS_EVIDENCE]->(ec)
            """, parameters={
                'ec_id': ec_id, 'pname': patient_name,
                'plt_nadir': plt_nadir,    'plt_below10': plt_below10,
                'hct_peak': hct_peak,      'hct_change_pct': hct_change_pct,
                'hflc_peak': hflc_peak,    'wbc_trend': wbc_trend,
                'ast_value': ast_value,    'alt_value': alt_value,
                'creatinine_value': creatinine_value,
                'ast_available': ast_available, 'critical_day': critical_day
            })

            # EvidenceCase → Symptoms
            for sym_name in patient_sym_list:
                db.execute_query("""
                    MATCH (ec:EvidenceCase {id: $ec_id})
                    MATCH (s:Symptom {name: $sname})
                    MERGE (ec)-[:HAS_SYMPTOM]->(s)
                """, parameters={'ec_id': ec_id, 'sname': sym_name})

            # EvidenceCase → Concepts
            db.execute_query("""
                MATCH (ec:EvidenceCase {id: $ec_id})
                OPTIONAL MATCH (ec)-[old:HAS_CONCEPT]->(:Concept)
                DELETE old
            """, parameters={'ec_id': ec_id})

            for concept_name in concept_names:
                db.execute_query("""
                    MATCH (ec:EvidenceCase {id: $ec_id})
                    MATCH (c:Concept {name: $cname})
                    MERGE (ec)-[:HAS_CONCEPT]->(c)
                """, parameters={'ec_id': ec_id, 'cname': concept_name})

            count += 1

        print(f"✅ Đã nạp {count} bệnh nhân + EvidenceCase thành công.")

        # ── SIMILAR_TO edges (Case-Based Reasoning) ───────────────────────
        print("🔗 Đang tạo SIMILAR_TO edges...")
        db.execute_query("""
            MATCH (p1:Patient)-[:HAS_SYMPTOM]->(s:Symptom)<-[:HAS_SYMPTOM]-(p2:Patient)
            WHERE p1.id < p2.id
            WITH p1, p2, count(DISTINCT s) AS shared
            WHERE shared >= 2
            MERGE (p1)-[:SIMILAR_TO {shared_symptoms: shared}]->(p2)
        """)
        print("✅ SIMILAR_TO edges đã được tạo.")

        print("Creating EvidenceCase -> MATCHES -> Diagnostic_Rule edges...")
        n_matches = refresh_evidence_rule_matches(db)
        print(f"Created/updated {n_matches} MATCHES edges.")

    finally:
        db.close()


if __name__ == "__main__":
    load_patients_to_graph(str(PROJECT_ROOT / "data" / "NghiencuuHFLC (1) - NghiencuuHFLC (1).csv"))
