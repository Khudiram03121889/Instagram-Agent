import fitz
import os

pdf_path = "CuriousMatterLab_Pipeline_Upgrade.pdf"
out_path = "scratch/pdf_text.txt"

os.makedirs("scratch", exist_ok=True)

doc = fitz.open(pdf_path)
text = []
for i, page in enumerate(doc):
    text.append(f"--- PAGE {i+1} ---")
    text.append(page.get_text())

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(text))

print("Successfully extracted PDF text to scratch/pdf_text.txt")
