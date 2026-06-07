import os

brain_dir = r"C:\Users\Dell\.gemini\antigravity\brain"
for root, dirs, files in os.walk(brain_dir):
    for f in files:
        if f == "transcript.jsonl":
            full_path = os.path.join(root, f)
            print(f"Transcript: {full_path} ({os.path.getsize(full_path)} bytes)")
