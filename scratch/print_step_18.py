import json

log_file = r"C:\Users\Dell\.gemini\antigravity\brain\54755bf9-29ec-4ff1-bbb2-e1a382275629\.system_generated\logs\transcript.jsonl"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("step_index") == 18:
            print(json.dumps(obj, indent=2))
            break
