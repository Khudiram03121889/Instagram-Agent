import os
import json

folder = r"C:\Users\Dell\AppData\Roaming\Antigravity\User\History\44df0810"
if os.path.exists(folder):
    print(f"Files in {folder}:")
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        print(f"  {f} - Size: {os.path.getsize(p)} bytes")
    # Read entries.json
    entries_path = os.path.join(folder, "entries.json")
    if os.path.exists(entries_path):
        with open(entries_path, "r", encoding="utf-8") as entry_f:
            print("\nentries.json content:")
            print(json.dumps(json.load(entry_f), indent=2))
