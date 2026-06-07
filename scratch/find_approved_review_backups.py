import os

search_paths = [
    r"d:\Python\Programs\instagram agent",
    r"d:\Python\Programs",
    r"C:\Users\Dell\.gemini\antigravity"
]

print("Searching for backup Python files containing 'approved-review'...")

found = []
for s_path in search_paths:
    if not os.path.exists(s_path):
        continue
    for root, dirs, files in os.walk(s_path):
        # Exclude .git and .venv
        if ".git" in dirs:
            dirs.remove(".git")
        if ".venv" in dirs:
            dirs.remove(".venv")
        if "node_modules" in dirs:
            dirs.remove("node_modules")
            
        for f in files:
            if f.endswith(".py") or f.endswith(".txt") or f.endswith(".bak"):
                full_path = os.path.join(root, f)
                try:
                    if os.path.getsize(full_path) > 20000: # only check larger files
                        with open(full_path, "r", encoding="utf-8", errors="ignore") as file:
                            content = file.read()
                            if "approved-review" in content or "approved_review" in content:
                                print(f"Found match: {full_path} ({os.path.getsize(full_path)} bytes)")
                                found.append(full_path)
                except Exception:
                    pass

print(f"Search complete. Found {len(found)} files.")
