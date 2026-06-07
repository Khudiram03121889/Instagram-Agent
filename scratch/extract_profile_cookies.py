import os
import shutil
import json
from playwright.sync_api import sync_playwright

def prepare_temp_profile():
    src_user_data = os.path.expandvars(r"%USERPROFILE%\AppData\Local\Google\Chrome\User Data")
    dest_user_data = os.path.abspath("scratch/temp_chrome_profile")
    
    if os.path.exists(dest_user_data):
        shutil.rmtree(dest_user_data)
        
    os.makedirs(dest_user_data, exist_ok=True)
    
    files_to_copy = [
        ("Local State", "Local State"),
        ("Default/Network/Cookies", "Default/Network/Cookies"),
        ("Default/Preferences", "Default/Preferences")
    ]
    
    for src_rel, dest_rel in files_to_copy:
        src_path = os.path.join(src_user_data, src_rel)
        dest_path = os.path.join(dest_user_data, dest_rel)
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            try:
                shutil.copyfile(src_path, dest_path)
                print(f"Copied {src_rel} to temp profile.")
            except Exception as e:
                print(f"Error copying {src_rel}: {e}")
        else:
            print(f"Source file not found: {src_path}")
            
    return dest_user_data

def main():
    print("Preparing temp Chrome profile directory...")
    temp_profile_dir = prepare_temp_profile()
    
    url = "https://labs.google/fx/tools/flow"
    auth_file = "auth.json"
    
    print("\nLaunching headless Playwright Chrome using temp profile...")
    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=temp_profile_dir,
                headless=True,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"]
            )
            
            page = context.new_page()
            print(f"Navigating to {url}...")
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            
            print("Checking if logged in...")
            page.wait_for_timeout(5000)
            
            current_url = page.url
            title = page.title()
            print(f"Current URL: {current_url}")
            print(f"Current Title: {title}")
            
            is_correct_url = current_url.split("?")[0].rstrip("/") in ["https://labs.google/fx/tools/flow", "https://labs.google/fx/tools/flow/"]
            is_redirecting = "google.com" in current_url or "google.com" in title.lower() or "loading" in title.lower()
            
            if is_correct_url and not is_redirecting and "sign in" not in title.lower() and "login" not in title.lower():
                print("\n[OK] Success! Detected active logged-in session in Chrome profile.")
                context.storage_state(path=auth_file)
                print(f"Saved active session to '{auth_file}'!")
            else:
                print("\n[FAIL] You are not signed in to Google Flow in your Chrome profile, or session expired.")
                
            context.close()
        except Exception as e:
            print(f"Error occurred during Playwright run: {e}")
            
    # Cleanup temp profile
    try:
        shutil.rmtree(temp_profile_dir)
        print("Cleaned up temp profile directory.")
    except Exception as e:
        print(f"Error cleaning up: {e}")

if __name__ == "__main__":
    main()
