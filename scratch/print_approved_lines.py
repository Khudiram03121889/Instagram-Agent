import os

views_dir = r"d:\Python\Programs\instagram agent\scratch\main_views"

for f in sorted(os.listdir(views_dir)):
    if f.endswith(".txt"):
        path = os.path.join(views_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file, 1):
                if "approved" in line.lower() or "review_state" in line.lower() or "review" in line.lower():
                    print(f"{f}:{idx}: {line.strip()}")
