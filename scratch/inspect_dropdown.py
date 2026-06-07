import os
from bs4 import BeautifulSoup

html_path = "outputs/Why do we close our eyes when we sneeze/flow_dropdown_failed.html"
if not os.path.exists(html_path):
    print("flow_dropdown_failed.html not found!")
    exit(1)

with open(html_path, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

print("=== SEARCHING FOR MODEL CONTROL ELEMENT ===")
# We know the text is "Omni Flash" or contains "arrow_drop_down"
for tag in soup.find_all(lambda t: t.name in ["div", "span", "button", "p"]):
    text = tag.get_text(" ", strip=True)
    if "Omni Flash" in text and "arrow_drop_down" in text:
        print(f"MATCH: <{tag.name}> text='{text}'")
        print(f"  Attrs: {tag.attrs}")
        print(f"  Parent: <{tag.parent.name}> attrs={tag.parent.attrs if tag.parent else None}")
        print(f"  HTML: {str(tag)[:200]}...")
        print("-" * 40)

print("\n=== SEARCHING FOR DROPDOWN OPTIONS ===")
for tag in soup.find_all(lambda t: t.name in ["div", "span", "button", "li", "option"]):
    text = tag.get_text(" ", strip=True)
    if text == "volume_up Omni Flash" or text == "Omni Flash":
        print(f"OPTION MATCH: <{tag.name}> text='{text}'")
        print(f"  Attrs: {tag.attrs}")
        print(f"  Parent: <{tag.parent.name}> attrs={tag.parent.attrs if tag.parent else None}")
        print(f"  HTML: {str(tag)[:200]}...")
        print("-" * 40)
