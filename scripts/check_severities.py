import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    print("=== SEVERITY NODES ===")
    res = db.execute_query("MATCH (s:Severity) RETURN s.name AS name")
    for r in res:
        print(f"Severity: {r['name']}")
finally:
    db.close()
