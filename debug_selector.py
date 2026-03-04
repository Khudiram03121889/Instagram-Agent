from playwright.sync_api import sync_playwright
import os

AUTH_FILE = "auth.json"
URL = "https://labs.google/fx/tools/flow"

def debug_selector():
    if not os.path.exists(AUTH_FILE):
        print(f"ERROR: {AUTH_FILE} not found. Run setup_auth.py first.")
        return

    print("🚀 Launching Chrome in DEBUG MODE...")
    print("📋 INSTRUCTIONS:")
    print("1. Wait for the browser and 'Playwright Inspector' window to open.")
    print("2. In the browser, navigate to a new project or open the menu manually.")
    print("3. In the Inspector window, click the 'Pick Locator' (target icon/cursor).")
    print("4. HOVER over the 'Text to Video' option in the browser.")
    print("5. The best selector will appear in the simplified locator box.")
    print("6. Copy that selector and tell me what it is!")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state=AUTH_FILE)
        page = context.new_page()
        
        print(f"🌐 Going to {URL}...")
        page.goto(URL)
        
        # Pause execution and open inspector
        page.pause()

if __name__ == "__main__":
    debug_selector()
