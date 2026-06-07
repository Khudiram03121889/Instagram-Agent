import re

file_path = r"d:\Python\Programs\instagram agent\outputs\Why do we close our eyes when we sneeze\flow_setup_failed.html"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()

# Find all HTML elements that have text containing Video or Image
# Let's search for "Video" in a case-sensitive regex inside tags:
for m in re.finditer(r'<[^>]*Video[^>]*>', html):
    pos = m.start()
    print(f"Tag with Video at {pos}: {html[pos:pos+150]}")

print("\nSearching for role='tab' or aria-haspopup:")
for m in re.finditer(r'aria-haspopup|role="tab"', html):
    pos = m.start()
    print(f"Match at {pos}: {html[pos-50:pos+150]}")
