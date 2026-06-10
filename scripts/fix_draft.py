import json
import re
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

draft_path = "data/draft_guidelines.json"

try:
    with open(draft_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("✅ Read draft_guidelines.json successfully.")
except Exception as e:
    print(f"❌ Failed to read draft_guidelines.json: {e}")
    sys.exit(1)

# Fix PUA character \uf06d (replace with standard Greek micro symbol μ)
pua_matches = len(re.findall('\uf06d', content))
if pua_matches > 0:
    content_fixed = content.replace('\uf06d', 'μ')
    print(f"🔧 Found {pua_matches} occurrences of PUA character '\\uf06d'. Replaced with standard 'μ'.")
else:
    content_fixed = content
    print("ℹ️ No PUA characters found.")

# Other replacements:
# Let's check for any duplicated spaces or formatting typos
content_fixed = re.sub(r' +', ' ', content_fixed)

# Try parsing as JSON to check syntax validity before saving
try:
    data = json.loads(content_fixed)
    print("✅ JSON syntax validation passed.")
    # Save the updated file
    with open(draft_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("💾 draft_guidelines.json updated successfully.")
except Exception as e:
    print(f"❌ JSON Syntax Error during fix validation: {e}")
    sys.exit(1)
