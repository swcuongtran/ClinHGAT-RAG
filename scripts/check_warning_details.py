import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(str(Path("c:/Users/Admin/Desktop/bệnh nhiệt đới")))

from clinical_cdss.core.database import Neo4jConnection

db = Neo4jConnection()
try:
    rule_name = "Sốt Dengue có dấu hiệu cảnh báo (Giai đoạn nguy hiểm)"
    print(f"=== DETAILS OF RULE: {rule_name} ===")
    res = db.execute_query("""
        MATCH (r:Diagnostic_Rule {name: $name})
        OPTIONAL MATCH (r)-[:DETERMINES]->(sev:Severity)
        RETURN r.name AS name, r.phase AS phase, sev.name AS severity
    """, {"name": rule_name})
    for r in res:
        print(r)
        
    print("\n=== SYMPTOMS/LABS OF RULE ===")
    res = db.execute_query("""
        MATCH (r:Diagnostic_Rule {name: $name})<-[:PART_OF_RULE]-(node)
        RETURN labels(node) AS labels, node.name AS name
    """, {"name": rule_name})
    for r in res:
        print(f"  Node: {r['name']} (Labels: {r['labels']})")
        
    print("\n=== CONCEPTS OF RULE ===")
    res = db.execute_query("""
        MATCH (r:Diagnostic_Rule {name: $name})<-[:LINKED_TO_RULE]-(c:Concept)
        RETURN c.name AS name
    """, {"name": rule_name})
    for r in res:
        print(f"  Concept: {r['name']}")
finally:
    db.close()
