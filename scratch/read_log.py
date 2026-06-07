import json

log_file = r"C:\Users\Dell\.gemini\antigravity\brain\54755bf9-29ec-4ff1-bbb2-e1a382275629\.system_generated\logs\transcript.jsonl"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("type") == "PLANNER_RESPONSE":
            if "tool_calls" in obj:
                for tc in obj["tool_calls"]:
                    if tc.get("name") == "view_file":
                        args = tc.get("args", {})
                        path = args.get("AbsolutePath", "")
                        if "main.py" in path:
                            print(f"Step {obj.get('step_index')}: VIEW_FILE lines {args.get('StartLine')} to {args.get('EndLine')}")
