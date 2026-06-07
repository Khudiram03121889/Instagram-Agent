import os

views_dir = r"d:\Python\Programs\instagram agent\scratch\main_views"
keywords = ["review", "check", "approved", "parse_cli_args", "dashboard"]

for f in sorted(os.listdir(views_dir)):
    if f.endswith(".txt"):
        path = os.path.join(views_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            matches = [k for k in keywords if k in content]
            if matches:
                print(f"File {f} matches: {matches}")
