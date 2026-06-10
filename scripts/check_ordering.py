import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    print("=== SEVERITY NAMES OF RULES ===")
    res = db.execute_query("""
        MATCH (r:Diagnostic_Rule)-[:DETERMINES]->(s:Severity)
        RETURN DISTINCT s.name AS name
    """)
    for r in res:
        name = r['name']
        # Compute priority
        if 'sốc' in name.lower() and ('nặng' in name.lower() or 'kéo dài' in name.lower() or 'thất bại' in name.lower()):
            prio = 5
        elif 'sốc' in name.lower():
            prio = 4
        elif 'cảnh báo' in name.lower() or 'chuyển độ' in name.lower():
            prio = 3
        elif 'sốt xuất huyết' in name.lower() or 'sốt dengue nặng' in name.lower():
            prio = 2
        elif 'sốt dengue' in name.lower() or 'sốt' in name.lower() or 'chăm sóc' in name.lower() or 'lo lắng' in name.lower():
            prio = 1
        elif 'chưa phân loại' in name.lower():
            prio = 0
        else:
            prio = 1
        print(f"Severity: {name} -> priority: {prio}")
finally:
    db.close()
