import re
import sys

# Force stdout to use backslashreplace for characters it can't print
sys.stdout.reconfigure(errors='backslashreplace')

file_path = r"d:\Python\Programs\instagram agent\outputs\Why do we close our eyes when we sneeze\flow_setup_failed.html"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()

print(f"HTML length: {len(html)}")

# Let's search for "Banana", "Nano", "What do you want", etc.
for term in ["Banana", "Nano", "What do you want", "radix", "menu", "settings", "x2"]:
    matches = [m.start() for m in re.finditer(term, html, re.IGNORECASE)]
    print(f"Matches for '{term}': {len(matches)}")
    if matches:
        start = max(0, matches[0] - 100)
        end = min(len(html), matches[0] + 100)
        snippet = html[start:end].replace('\n', ' ')
        print(f"  Snippet: {snippet}")

# Let's find all button tags using regex
buttons = re.findall(r'<button[^>]*>.*?</button>', html, re.DOTALL | re.IGNORECASE)
print(f"\nFound {len(buttons)} button elements (regex):")
for i, btn in enumerate(buttons[:100]):
    clean_btn = re.sub(r'\s+', ' ', btn).strip()
    if len(clean_btn) > 300:
        clean_btn = clean_btn[:150] + " ... " + clean_btn[-150:]
    print(f"  {i}: {clean_btn}")
