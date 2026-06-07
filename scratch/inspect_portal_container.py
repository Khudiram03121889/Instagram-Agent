import re

file_path = r"d:\Python\Programs\instagram agent\outputs\Why do we close our eyes when we sneeze\flow_setup_failed.html"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()

# Search for where "Image" and "Video" appear close together in the HTML.
matches = []
for m in re.finditer(r'Video', html):
    pos = m.start()
    start = max(0, pos - 500)
    end = min(len(html), pos + 500)
    snippet = html[start:end].replace('\n', ' ')
    if "Image" in snippet:
        print(f"Match near {pos}: ... {snippet[:400]} ...")
        print("-" * 50)
