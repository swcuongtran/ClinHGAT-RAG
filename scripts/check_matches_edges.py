import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    pid = "nguyenductuyen"
    print("=== ALL MATCHES EDGES FROM EVIDENCE CASE ===")
    res = db.execute_query("""
        MATCH (p:Patient {id: $pid})-[:HAS_EVIDENCE]->(ec:EvidenceCase)-[m:MATCHES]->(r:Diagnostic_Rule)
        OPTIONAL MATCH (r)-[:DETERMINES]->(s:Severity)
        RETURN r.name AS rule_name, s.name AS severity, m.coverage_score AS score, 
               m.sym_match AS sym_m, m.sym_total AS sym_t, 
               m.concept_match AS con_m, m.concept_total AS con_t
        ORDER BY score DESC, rule_name ASC
    """, {"pid": pid})
    for r in res:
        print(f"Rule: {r['rule_name']} | Severity: {r['severity']}")
        print(f"  Score: {r['score']:.4f}")
        print(f"  Symptoms: {r['sym_m']}/{r['sym_t']}")
        print(f"  Concepts: {r['con_m']}/{r['con_t']}")
        print("-" * 50)
finally:
    db.close()
