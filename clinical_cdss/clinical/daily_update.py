from typing import Optional

from clinical_cdss.core.database import Neo4jConnection
from clinical_cdss.etl.patients import compute_slope, refresh_evidence_rule_matches


DAILY_DERIVED_CONCEPTS = {
    "Bệnh nhi",
    "Người lớn",
    "Giảm tiểu cầu",
    "Giảm tiểu cầu nặng",
    "Giảm tiểu cầu rất nặng",
    "Giảm tiểu cầu nguy kịch",
    "Cô đặc máu",
    "HCT tăng >=20%",
    "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)",
    "Nadir tiểu cầu trong pha nguy hiểm",
    "Bạch cầu giảm dần",
    "Bạch cầu tăng nhanh",
    "HFLC tăng",
    "HFLC tăng cao",
    "Tăng men gan",
    "Tổn thương gan nặng / suy gan cấp",
    "Tổn thương thận cấp / giảm tưới máu thận",
}


def _thresholds(age: float, gender: int):
    is_child = age <= 15
    is_male = gender == 1
    return 38.0 if is_child else (45.0 if is_male else 40.0)


def upsert_daily_record(
    patient_id: str,
    hospital_day: int,
    disease_day: Optional[int] = None,
    wbc: Optional[float] = None,
    plt: Optional[float] = None,
    hct: Optional[float] = None,
    hflc: Optional[float] = None,
):
    """Insert/update one daily observation and refresh derived graph evidence."""
    db = Neo4jConnection()
    try:
        patient_rows = db.execute_query("""
            MATCH (p:Patient {id: $pid})
            RETURN p.age AS age,
                   p.gender AS gender,
                   p.admission_day AS admission_day
        """, {"pid": patient_id})
        if not patient_rows:
            return {"error": f"Patient not found: {patient_id}"}

        patient = patient_rows[0]
        if disease_day is None:
            admission_day = patient["admission_day"] or 1
            disease_day = int(admission_day) + int(hospital_day) - 1

        db.execute_query("""
            MATCH (p:Patient {id: $pid})
            MERGE (dr:DailyRecord {id: p.id + '_day_' + toString($hospital_day)})
            SET dr.day = $hospital_day,
                dr.disease_day = $disease_day,
                dr.wbc = $wbc,
                dr.plt = $plt,
                dr.hct = $hct,
                dr.hflc = $hflc
            MERGE (p)-[:HAS_RECORD]->(dr)
        """, {
            "pid": patient_id,
            "hospital_day": int(hospital_day),
            "disease_day": int(disease_day),
            "wbc": wbc,
            "plt": plt,
            "hct": hct,
            "hflc": hflc,
        })

        summary = refresh_patient_evidence(db, patient_id)
        refresh_evidence_rule_matches(db, evidence_id=f"ec_{patient_id}")
        return summary
    finally:
        db.close()


def refresh_patient_evidence(db: Neo4jConnection, patient_id: str):
    rows = db.execute_query("""
        MATCH (p:Patient {id: $pid})
        OPTIONAL MATCH (p)-[:HAS_RECORD]->(dr:DailyRecord)
        WITH p, dr
        ORDER BY dr.disease_day, dr.day
        RETURN p.age AS age,
               p.gender AS gender,
               p.ast_value AS ast_value,
               p.alt_value AS alt_value,
               p.creatinine_value AS creatinine_value,
               collect({
                   disease_day: dr.disease_day,
                   wbc: dr.wbc,
                   plt: dr.plt,
                   hct: dr.hct,
                   hflc: dr.hflc
               }) AS records
    """, {"pid": patient_id})
    if not rows:
        return {"error": f"Patient not found: {patient_id}"}

    row = rows[0]
    records = [r for r in row["records"] if r["disease_day"] is not None]
    plt_vals = [r["plt"] for r in records if r["plt"] is not None]
    hct_vals = [r["hct"] for r in records if r["hct"] is not None and r["hct"] <= 100]
    hflc_vals = [r["hflc"] for r in records if r["hflc"] is not None]
    wbc_pairs = [(r["disease_day"], r["wbc"]) for r in records if r["wbc"] is not None]

    plt_nadir = min(plt_vals) if plt_vals else None
    plt_below10 = any(value < 10 for value in plt_vals)
    hct_peak = max(hct_vals) if hct_vals else None
    hflc_peak = max(hflc_vals) if hflc_vals else None
    wbc_trend = compute_slope(wbc_pairs)
    hct_baseline = hct_vals[0] if hct_vals else None
    hct_change_pct = (
        (hct_peak - hct_baseline) / hct_baseline * 100
        if hct_peak and hct_baseline and hct_baseline > 0
        else 0.0
    )
    plt_with_day = [(r["disease_day"], r["plt"]) for r in records if r["plt"] is not None]
    critical_day = min(plt_with_day, key=lambda item: item[1])[0] if plt_with_day else None

    age = float(row["age"] or 18.0)
    gender = int(row["gender"] or 2)
    ast_value = float(row["ast_value"] or 0.0)
    alt_value = float(row["alt_value"] or 0.0)
    creatinine_value = float(row["creatinine_value"] or 0.0)
    hct_threshold = _thresholds(age, gender)
    has_low_plt = plt_nadir is not None and plt_nadir < 100
    has_high_hct = hct_peak is not None and hct_peak > hct_threshold
    is_child = age <= 15

    db.execute_query("""
        MATCH (p:Patient {id: $pid})
        MERGE (ec:EvidenceCase {id: 'ec_' + p.id})
        SET p.plt_nadir = $plt_nadir,
            p.plt_below10 = $plt_below10,
            p.hct_peak = $hct_peak,
            p.hct_change_pct = $hct_change_pct,
            p.hflc_peak = $hflc_peak,
            p.wbc_trend = $wbc_trend,
            ec.plt_nadir = $plt_nadir,
            ec.plt_below10 = $plt_below10,
            ec.hct_peak = $hct_peak,
            ec.hct_change_pct = $hct_change_pct,
            ec.hflc_peak = $hflc_peak,
            ec.wbc_trend = $wbc_trend,
            ec.ast_value = $ast_value,
            ec.alt_value = $alt_value,
            ec.creatinine_value = $creatinine_value,
            ec.critical_day = $critical_day
        MERGE (p)-[:HAS_EVIDENCE]->(ec)
    """, {
        "pid": patient_id,
        "plt_nadir": plt_nadir,
        "plt_below10": plt_below10,
        "hct_peak": hct_peak,
        "hct_change_pct": hct_change_pct,
        "hflc_peak": hflc_peak,
        "wbc_trend": wbc_trend,
        "ast_value": ast_value,
        "alt_value": alt_value,
        "creatinine_value": creatinine_value,
        "critical_day": critical_day,
    })

    db.execute_query("""
        MATCH (p:Patient {id: $pid})
        OPTIONAL MATCH (p)-[old:HAS_CONDITION]->(c:Concept)
        WHERE c.name IN $derived_concepts
        DELETE old
        WITH p
        MATCH (p)-[:HAS_EVIDENCE]->(ec:EvidenceCase)
        OPTIONAL MATCH (ec)-[old_ec:HAS_CONCEPT]->(ec_c:Concept)
        WHERE ec_c.name IN $derived_concepts
        DELETE old_ec
    """, {
        "pid": patient_id,
        "derived_concepts": sorted(DAILY_DERIVED_CONCEPTS),
    })

    concept_names = []

    def add(enabled, name):
        if enabled and name not in concept_names:
            concept_names.append(name)

    add(is_child, "Bệnh nhi")
    add(not is_child, "Người lớn")
    add(has_low_plt, "Giảm tiểu cầu")
    add(plt_nadir is not None and plt_nadir < 50, "Giảm tiểu cầu nặng")
    add(plt_nadir is not None and plt_nadir < 20, "Giảm tiểu cầu rất nặng")
    add(plt_nadir is not None and plt_nadir < 10, "Giảm tiểu cầu nguy kịch")
    add(has_high_hct, "Cô đặc máu")
    add(hct_change_pct >= 20, "HCT tăng >=20%")
    add(has_low_plt and has_high_hct, "Dấu hiệu cảnh báo Sốc (HCT tăng, PLT giảm)")
    add(critical_day is not None and 4 <= int(critical_day) <= 7, "Nadir tiểu cầu trong pha nguy hiểm")
    add(wbc_trend < -0.5, "Bạch cầu giảm dần")
    add(wbc_trend > 1.0, "Bạch cầu tăng nhanh")
    add(hflc_peak is not None and hflc_peak >= 1.0, "HFLC tăng")
    add(hflc_peak is not None and hflc_peak >= 2.0, "HFLC tăng cao")
    add(ast_value >= 200 or alt_value >= 200, "Tăng men gan")
    add(ast_value >= 1000 or alt_value >= 1000, "Tổn thương gan nặng / suy gan cấp")
    add(creatinine_value >= (70.0 if is_child else 110.0), "Tổn thương thận cấp / giảm tưới máu thận")

    for concept_name in concept_names:
        db.execute_query("""
            MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)
            MERGE (c:Concept {name: $concept_name})
            MERGE (p)-[:HAS_CONDITION]->(c)
            MERGE (ec)-[:HAS_CONCEPT]->(c)
        """, {"pid": patient_id, "concept_name": concept_name})

    return {
        "patient_id": patient_id,
        "plt_nadir": plt_nadir,
        "hct_peak": hct_peak,
        "hct_change_pct": hct_change_pct,
        "hflc_peak": hflc_peak,
        "wbc_trend": wbc_trend,
        "critical_day": critical_day,
        "concepts": concept_names,
    }
