import os
import json

cursor_history_dir = os.path.expandvars(r"%APPDATA%\Cursor\User\History")
print(f"Checking Cursor history in: {cursor_history_dir}")

if os.path.exists(cursor_history_dir):
    found = []
    # Walk through folders
    for folder in os.listdir(cursor_history_dir):
        folder_path = os.path.join(cursor_history_dir, folder)
        if os.path.isdir(folder_path):
            entries_path = os.path.join(folder_path, "entries.json")
            if os.path.exists(entries_path):
                try:
                    with open(entries_path, "r", encoding="utf-8") as entry_f:
                        entry_data = json.load(entry_f)
                        resource = entry_data.get("resource", "")
                        if "main.py" in resource:
                            # Print all versions in this history
                            for entry in entry_data.get("entries", []):
                                file_id = entry.get("id", "")
                                version_path = os.path.join(folder_path, file_id)
                                if os.path.exists(version_path):
                                    mtime = os.path.getmtime(version_path)
                                    found.append((version_path, resource, mtime, os.path.getsize(version_path)))
                except Exception as e:
                    pass
    
    # Sort by mtime descending
    found.sort(key=lambda x: x[2], reverse=True)
    print(f"Found {len(found)} main.py history entries:")
    for path, res, mtime, size in found:
        print(f"  Path: {path}\n  Resource: {res}\n  Size: {size} bytes\n  Time: {mtime}\n")
else:
    print("Cursor history directory not found.")
