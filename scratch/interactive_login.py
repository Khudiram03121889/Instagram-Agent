import time
from playwright.sync_api import sync_playwright

def main():
    url = "https://labs.google/fx/tools/flow"
    auth_file = "auth.json"
    
    print("\n" + "="*60)
    print("KEY INTERACTIVE LOGIN HELPER")
    print("="*60)
    print("This helper will open a visible Chrome window to verify or complete your login.")
    print("If you are already logged in, it will save the session automatically and close.")
    print("If not, please sign in. Once you reach the Flow editor, it will save and close.\n")
    
    with sync_playwright() as playwright:
        try:
            print("Launching Chrome...")
            browser = playwright.chromium.launch(
                headless=False,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"]
            )
            
            # Load existing cookies if present, to avoid re-login if they are valid
            import os
            if os.path.exists(auth_file):
                print(f"Loading existing session from {auth_file}...")
                context = browser.new_context(storage_state=auth_file)
            else:
                context = browser.new_context()
                
            page = context.new_page()
            print(f"Navigating to {url}...")
            page.goto(url, timeout=120000, wait_until="domcontentloaded")
            
            print("\nMonitoring page status. Please complete login in the opened browser window if needed...")
            
            # Poll every 2 seconds for up to 5 minutes (150 iterations)
            success = False
            for i in range(150):
                page.wait_for_timeout(2000)
                try:
                    current_url = page.url
                    title = page.title()
                    
                    # If we are on the Flow editor page (which is not a sign-in or consent page)
                    is_correct_url = current_url.split("?")[0].rstrip("/") in ["https://labs.google/fx/tools/flow", "https://labs.google/fx/tools/flow/"]
                    is_redirecting = "google.com" in current_url or "google.com" in title.lower() or "loading" in title.lower()
                    if is_correct_url and not is_redirecting and "sign in" not in title.lower() and "login" not in title.lower():
                        print(f"\nSuccessful login detected!")
                        print(f"   URL: {current_url}")
                        print(f"   Title: {title}")
                        
                        # Save the storage state
                        context.storage_state(path=auth_file)
                        print(f"Session saved successfully to '{auth_file}'!")
                        success = True
                        break
                    else:
                        # Print status update occasionally
                        if i % 5 == 0:
                            print(f"   [Status] URL: {current_url} | Title: {title}")
                except Exception as e:
                    print(f"   [Error reading page status]: {e}")
                    
            if not success:
                print("\nTimeout: Login was not completed within 5 minutes.")
                
            browser.close()
            
        except Exception as outer_err:
            print(f"\nBrowser error: {outer_err}")

if __name__ == "__main__":
    main()
