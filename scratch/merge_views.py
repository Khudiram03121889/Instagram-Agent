import os
import re

views_dir = r"d:\Python\Programs\instagram agent\scratch\main_views"
reconstructed = {}

# Regex to match "<line_number>: <code_line>"
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
                    # Store line. If there are duplicates, we keep the code (they should be identical anyway)
                    reconstructed[line_num] = code

# Let's print stats
sorted_lines = sorted(reconstructed.keys())
print(f"Reconstructed {len(sorted_lines)} lines of main.py")
if sorted_lines:
    print(f"Line range: {sorted_lines[0]} to {sorted_lines[-1]}")
    
    # Find gaps
    gaps = []
    start_gap = None
    for i in range(1, sorted_lines[-1] + 1):
        if i not in reconstructed:
            if start_gap is None:
                start_gap = i
        else:
            if start_gap is not None:
                gaps.append((start_gap, i - 1))
                start_gap = None
    if start_gap is not None:
        gaps.append((start_gap, sorted_lines[-1]))
        
    print(f"Found {len(gaps)} gaps:")
    for start, end in gaps:
        print(f"  Gap: {start} to {end} ({end - start + 1} lines)")
