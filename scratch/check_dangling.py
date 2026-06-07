import subprocess

blobs = [
    "96f54cd217d658b90062b8880b24a4f833dac22e",
    "99f52d63ddb178356fa585db1b3a641e45b8b9bf",
    "bcb658f804443f715d874de699d4534dac2fdb91",
    "5db778fe11dacfffdeecc152aad777b08e12eb93",
    "2df8fb8a43884c2b852ab494c19adaf964c30065",
    "04fbb8ae53e917dfdd1612d93520d642a5f69fec",
    "e43e3e7a8690aaf4b5af680fc15d3d961af1ab91",
    "29df57fe80c584d26f0e5e6f5b0ee3ed188a075e",
    "a71f37e1efc0c086698b4c2b42317c4b3800671b"
]

for blob in blobs:
    try:
        # Get content size of blob
        size = int(subprocess.check_output(["git", "cat-file", "-s", blob]).decode().strip())
        print(f"Blob {blob}: {size} bytes")
        if size > 50000: # If size is large, show preview
            content = subprocess.check_output(["git", "cat-file", "-p", blob]).decode("utf-8", errors="ignore")
            print(f"--- PREVIEW OF {blob} ---")
            print(content[:300])
            print("...\n")
    except Exception as e:
        print(f"Error checking {blob}: {e}")
