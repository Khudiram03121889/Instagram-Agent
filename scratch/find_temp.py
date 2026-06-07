import os

temp_dir = os.path.expandvars(r"%TEMP%")
print(f"Searching for main.py in: {temp_dir}")
found = []
for root, dirs, files in os.walk(temp_dir):
    for f in files:
        if "main.py" in f:
            full_path = os.path.join(root, f)
            try:
                size = os.path.getsize(full_path)
                found.append((full_path, size))
            except Exception:
                pass

print(f"Found {len(found)} files in Temp:")
for path, size in sorted(found, key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {path} ({size} bytes)")
