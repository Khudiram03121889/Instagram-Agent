import subprocess
import re

print("Running git fsck...")
output = subprocess.check_output(["git", "fsck", "--lost-found"]).decode("utf-8", errors="ignore")

commits = []
for line in output.splitlines():
    if "dangling commit" in line:
        commit_id = line.split()[-1]
        commits.append(commit_id)

print(f"Found {len(commits)} dangling commits. Checking each commit's main.py file...")

for commit in commits:
    try:
        # Check if main.py exists in this commit
        lines = subprocess.check_output(["git", "show", f"{commit}:main.py"]).decode("utf-8", errors="ignore").splitlines()
        num_lines = len(lines)
        print(f"Commit {commit}: main.py has {num_lines} lines")
        if num_lines == 2152:
            print(f"🎉 FOUND IT! Commit: {commit}")
    except Exception:
        pass
