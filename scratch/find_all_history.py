import os
import json

history_dirs = [
    r"C:\Users\Dell\AppData\Roaming\Antigravity\User\History",
    r"C:\Users\Dell\AppData\Roaming\Antigravity IDE\User\History",
    r"C:\Users\Dell\AppData\Roaming\Qoder\User\History",
    r"C:\Users\Dell\AppData\Roaming\Trae\User\History"
]

found = []

for hdir in history_dirs:
    print(f"Checking history in: {hdir}")
    if os.path.exists(hdir):
        for folder in os.listdir(hdir):
            folder_path = os.path.join(hdir, folder)
            if os.path.isdir(folder_path):
                entries_path = os.path.join(folder_path, "entries.json")
                if os.path.exists(entries_path):
                    try:
                        with open(entries_path, "r", encoding="utf-8") as entry_f:
                            entry_data = json.load(entry_f)
                            resource = entry_data.get("resource", "")
                            if "main.py" in resource:
                                print(f"  Found main.py in resource: {resource} ({hdir})")
                                for entry in entry_data.get("entries", []):
                                    file_id = entry.get("id", "")
                                    version_path = os.path.join(folder_path, file_id)
                                    if os.path.exists(version_path):
                                        mtime = os.path.getmtime(version_path)
                                        size = os.path.getsize(version_path)
                                        found.append((version_path, resource, mtime, size, hdir))
                    except Exception as e:
                        pass
    else:
        print(f"  Path does not exist: {hdir}")

# Sort found by mtime descending
found.sort(key=lambda x: x[2], reverse=True)
print(f"\nTotal main.py entries found: {len(found)}")
for path, res, mtime, size, hdir in found[:20]:
    print(f"  Path: {path}\n  Resource: {res}\n  Size: {size} bytes\n  Time: {mtime}\n  From: {hdir}\n")
