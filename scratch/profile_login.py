import os
import sys
from playwright.sync_api import sync_playwright

def main():
    url = "https://labs.google/fx/tools/flow"
    auth_file = "auth.json"
    
    # Load Chrome profile settings from .env
    chrome_user_data_dir = r"C:\Users\Dell\AppData\Local\Google\Chrome\User Data"
    chrome_profile = "Default"
    
    print("\n" + "="*60)
    print("CHROME PROFILE SESSION EXTRACTOR")
    print("="*60)
    print("This script will launch Chrome using your personal Default profile.")
    print("IMPORTANT: You MUST close all open Google Chrome windows before running this,")
    print("otherwise Chrome will prevent Playwright from accessing your profile.\n")
    
    if not os.path.isdir(chrome_user_data_dir):
        print(f"Error: Chrome user data directory not found: {chrome_user_data_dir}")
        return

    with sync_playwright() as playwright:
        try:
            print("Launching Chrome with your Default profile...")
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_user_data_dir,
                headless=False,
                channel="chrome",
                args=[
                    f"--profile-directory={chrome_profile}",
                    "--disable-blink-features=AutomationControlled"
                ]
            )
            
            page = context.new_page()
            print(f"Navigating to {url}...")
            page.goto(url, timeout=120000, wait_until="domcontentloaded")
            
            print("\nChecking if logged in...")
            page.wait_for_timeout(5000)
            
            current_url = page.url
            title = page.title()
            
            print(f"Current URL: {current_url}")
            print(f"Current Title: {title}")
            
            is_correct_url = current_url.split("?")[0].rstrip("/") in ["https://labs.google/fx/tools/flow", "https://labs.google/fx/tools/flow/"]
            is_redirecting = "google.com" in current_url or "google.com" in title.lower() or "loading" in title.lower()
            
            if is_correct_url and not is_redirecting and "sign in" not in title.lower() and "login" not in title.lower():
                print("\n🎉 Success! Detected active logged-in session.")
                context.storage_state(path=auth_file)
                print(f"Saved active session to '{auth_file}'!")
            else:
                print("\n⚠️ You are not signed in on this profile, or the session was blocked.")
                print("Please sign in manually in the opened Chrome window.")
                print("Monitoring page status for up to 3 minutes...")
                
                success = False
                for i in range(90):
                    page.wait_for_timeout(2000)
                    try:
                        current_url = page.url
                        title = page.title()
                        
                        is_correct_url = current_url.split("?")[0].rstrip("/") in ["https://labs.google/fx/tools/flow", "https://labs.google/fx/tools/flow/"]
                        is_redirecting = "google.com" in current_url or "google.com" in title.lower() or "loading" in title.lower()
                        
                        if is_correct_url and not is_redirecting and "sign in" not in title.lower() and "login" not in title.lower():
                            print("\n🎉 Successful login detected!")
                            context.storage_state(path=auth_file)
                            print(f"Saved session to '{auth_file}'!")
                            success = True
                            break
                    except Exception as e:
                        pass
                
                if not success:
                    print("\nTimeout: Active session could not be established.")
                    
            context.close()
            
        except Exception as e:
            print(f"\nError: {e}")
            print("\nIf you got a 'locked' or 'in use' error, make sure all Chrome instances are fully closed.")

if __name__ == "__main__":
    main()
