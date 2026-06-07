import os

repo_dir = r"d:\Python\Programs\instagram agent"
print(f"Searching for .idea or other history folders in: {repo_dir}")

for item in os.listdir(repo_dir):
    full_path = os.path.join(repo_dir, item)
    if os.path.isdir(full_path):
        print(f"  Dir: {item}")
