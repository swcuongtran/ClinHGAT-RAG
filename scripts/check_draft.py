import json
import re
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

draft_path = "data/draft_guidelines.json"

with open(draft_path, "r", encoding="utf-8") as f:
    content = f.read()

print("Scanning for PUA character \\uf06d (code 61549):")
matches = [m.start() for m in re.finditer('\uf06d', content)]
for idx in matches:
    start = max(0, idx - 40)
    end = min(len(content), idx + 40)
    snippet = content[start:end].replace('\n', ' ')
    print(f"  Index {idx}: ...{snippet}...")
