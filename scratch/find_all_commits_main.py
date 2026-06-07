import subprocess

print("Searching all commits for main.py line counts...")
output = subprocess.check_output(["git", "log", "--format=%H %s"]).decode("utf-8", errors="ignore")

for line in output.splitlines():
    parts = line.split(" ", 1)
    commit = parts[0]
    subject = parts[1] if len(parts) > 1 else ""
    try:
        content = subprocess.check_output(["git", "show", f"{commit}:main.py"]).decode("utf-8", errors="ignore")
        num_lines = len(content.splitlines())
        print(f"Commit {commit[:8]} ({subject[:30]}): {num_lines} lines")
        if num_lines > 2000:
            print(f"🎉 FOUND! {commit} has {num_lines} lines")
    except Exception:
        pass
