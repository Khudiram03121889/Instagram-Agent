import json

log_file = r"C:\Users\Dell\.gemini\antigravity\brain\54755bf9-29ec-4ff1-bbb2-e1a382275629\.system_generated\logs\transcript.jsonl"

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("type") == "VIEW_FILE" and "main.py" in obj.get("content", ""):
            step = obj.get("step_index")
            content = obj.get("content", "")
            if "1353:" in content:
                print(f"Step {step} length: {len(content)}")
                print("First 500 chars:")
                print(content[:500])
                print("\nLast 500 chars:")
                print(content[-500:])
                # Check if there is "<truncated" in the content
                if "<truncated" in content:
                    print("\n⚠️ The content is indeed truncated inside the transcript!")
                else:
                    print("\n🎉 The content is NOT truncated in the transcript!")
