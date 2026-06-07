import os

for f in os.listdir("scratch"):
    if f.startswith("step_") and f.endswith("_content.txt"):
        print(f"=== {f} ===")
        with open(os.path.join("scratch", f), "r", encoding="utf-8") as file:
            print(file.read(300))
            print("...\n")
