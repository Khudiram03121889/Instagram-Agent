import re

html_path = r"d:\Python\Programs\instagram agent\outputs\Why do we close our eyes when we sneeze\flow_setup_failed.html"

with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html = f.read()

print(f"HTML length: {len(html)}")

# Let's check which buttons are present in the failed HTML
buttons = re.findall(r'<button[^>]*>.*?</button>', html, re.DOTALL | re.IGNORECASE)
print(f"\nFound {len(buttons)} button elements in failed HTML:")
for i, btn in enumerate(buttons):
    clean_btn = re.sub(r'\s+', ' ', btn).strip()
    if len(clean_btn) > 300:
        clean_btn = clean_btn[:150] + " ... " + clean_btn[-150:]
    # Check if this button matches any of our selectors
    is_pop = "aria-haspopup" in clean_btn
    print(f"  {i}: popup={is_pop} -> {clean_btn}")
