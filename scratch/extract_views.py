import json
import os

log_file = r"C:\Users\Dell\.gemini\antigravity\brain\54755bf9-29ec-4ff1-bbb2-e1a382275629\.system_generated\logs\transcript.jsonl"
output_dir = r"d:\Python\Programs\instagram agent\scratch\main_views"
os.makedirs(output_dir, exist_ok=True)

print(f"Reading {log_file}...")

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("type") == "VIEW_FILE" and "main.py" in obj.get("content", ""):
            # Let's try to parse the lines and step index
            step = obj.get("step_index")
            content = obj.get("content", "")
            
            # Save the content of this step view
            out_file = os.path.join(output_dir, f"step_{step}.txt")
            with open(out_file, "w", encoding="utf-8") as out_f:
                out_f.write(content)
            print(f"Extracted step {step} view to {out_file} (length: {len(content)})")
