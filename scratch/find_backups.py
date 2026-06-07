import os

app_data = r"C:\Users\Dell\.gemini\antigravity"
print(f"Searching for main.py in {app_data}...")

for root, dirs, files in os.walk(app_data):
    for f in files:
        if "main.py" in f or "main" in f:
            full_path = os.path.join(root, f)
            print(f"Found: {full_path} ({os.path.getsize(full_path)} bytes)")
