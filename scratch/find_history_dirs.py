import os

appdata = os.path.expandvars(r"%APPDATA%")
print(f"Searching for History folders in {appdata}...")

for root, dirs, files in os.walk(appdata):
    # Check if 'History' is in the dirs list (doing this keeps it fast and avoids walking deep unnecessarily)
    if "History" in dirs:
        full_path = os.path.join(root, "History")
        print(f"Found History folder: {full_path}")
        # Stop walking down this folder to prevent infinite recursion or redundancy
        dirs.remove("History")
