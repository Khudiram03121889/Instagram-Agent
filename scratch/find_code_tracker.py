import os

code_tracker_dir = r"C:\Users\Dell\.gemini\antigravity\code_tracker"
print(f"Listing all files in: {code_tracker_dir}")

for root, dirs, files in os.walk(code_tracker_dir):
    for f in files:
        full_path = os.path.join(root, f)
        size = os.path.getsize(full_path)
        print(f"  {full_path} ({size} bytes)")
