import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    pid = "nguyenductuyen"
    print("=== PATIENT NODES & PROPERTIES ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid}) RETURN properties(p) AS props", {"pid": pid})
    for r in res:
        print(r['props'])
        
    print("\n=== PATIENT SYMPTOMS ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS name", {"pid": pid})
    for r in res:
        print(f"  Symptom: {r['name']}")
        
    print("\n=== PATIENT CONCEPTS ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_CONDITION]->(c:Concept) RETURN c.name AS name", {"pid": pid})
    for r in res:
        print(f"  Concept: {r['name']}")
        
    print("\n=== DAILY RECORDS ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_RECORD]->(dr:DailyRecord) RETURN properties(dr) AS props ORDER BY dr.disease_day", {"pid": pid})
    for r in res:
        print(f"  Day {r['props'].get('disease_day')}: plt={r['props'].get('plt')}, hct={r['props'].get('hct')}, wbc={r['props'].get('wbc')}, hflc={r['props'].get('hflc')}")
        
    print("\n=== EVIDENCE CASES ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase) RETURN properties(ec) AS props", {"pid": pid})
    for r in res:
        print(r['props'])
        
    print("\n=== EVIDENCE CASE SYMPTOMS ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)-[:HAS_SYMPTOM]->(s:Symptom) RETURN s.name AS name", {"pid": pid})
    for r in res:
        print(f"  Symptom: {r['name']}")
        
    print("\n=== EVIDENCE CASE CONCEPTS ===")
    res = db.execute_query("MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)-[:HAS_CONCEPT]->(c:Concept) RETURN c.name AS name", {"pid": pid})
    for r in res:
        print(f"  Concept: {r['name']}")

finally:
    db.close()
