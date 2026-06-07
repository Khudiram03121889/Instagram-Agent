import os
import re
import sys

# Force UTF-8 stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

views_dir = r"d:\Python\Programs\instagram agent\scratch\main_views"
reconstructed = {}
line_re = re.compile(r"^(\d+):\s?(.*)$")

for f in os.listdir(views_dir):
    if f.endswith(".txt"):
        path = os.path.join(views_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                match = line_re.match(line)
                if match:
                    line_num = int(match.group(1))
                    code = match.group(2)
                    reconstructed[line_num] = code

print("Reconstructed main() start:")
for idx in range(700, 830):
    if idx in reconstructed:
        print(f"{idx}: {reconstructed[idx]}")
