import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    print("=== RULES CONTAINING 'cảnh báo' ===")
    res = db.execute_query("""
        MATCH (r:Diagnostic_Rule)
        WHERE r.name CONTAINS 'cảnh báo' OR r.name CONTAINS 'Cảnh báo'
        RETURN r.name AS name, r.phase AS phase
    """)
    for r in res:
        print(f"Rule: {r['name']} | Phase: {r['phase']}")
finally:
    db.close()
