import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    rules = ["Cảnh báo (Sốc Dengue)", "Sốt Dengue có dấu hiệu cảnh báo (Giai đoạn nguy hiểm)"]
    for rname in rules:
        print(f"\n==========================================")
        print(f"RULE: {rname}")
        print(f"==========================================")
        res = db.execute_query("""
            MATCH (r:Diagnostic_Rule {name: $name})
            OPTIONAL MATCH (r)-[:DETERMINES]->(sev:Severity)
            RETURN r.name AS name, r.phase AS phase, sev.name AS severity
        """, {"name": rname})
        for r in res:
            print(f"Details: {r}")
            
        print("\n--- Symptoms (PART_OF_RULE) ---")
        res = db.execute_query("""
            MATCH (r:Diagnostic_Rule {name: $name})<-[:PART_OF_RULE]-(s:Symptom)
            RETURN s.name AS name
        """, {"name": rname})
        for r in res:
            print(f"  Symptom: {r['name']}")
            
        print("\n--- Lab Tests (PART_OF_RULE) ---")
        res = db.execute_query("""
            MATCH (r:Diagnostic_Rule {name: $name})<-[:PART_OF_RULE]-(l:LabTest)
            OPTIONAL MATCH (l)-[:MAPS_TO]->(c:Concept)
            RETURN l.name AS name, c.name AS concept
        """, {"name": rname})
        for r in res:
            print(f"  LabTest: {r['name']} -> Concept: {r['concept']}")
            
        print("\n--- Concepts (LINKED_TO_RULE) ---")
        res = db.execute_query("""
            MATCH (r:Diagnostic_Rule {name: $name})<-[:LINKED_TO_RULE]-(c:Concept)
            RETURN c.name AS name
        """, {"name": rname})
        for r in res:
            print(f"  Concept: {r['name']}")
finally:
    db.close()
