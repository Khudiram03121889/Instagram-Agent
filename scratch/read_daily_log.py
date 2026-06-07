with open("daily_log.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines in daily_log.txt: {len(lines)}")
for i, line in enumerate(lines):
    if "check_login" in line:
        print(f"Line {i}: {line.strip()[:100]}")
