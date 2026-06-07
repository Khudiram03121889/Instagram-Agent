import os
from playwright.sync_api import sync_playwright

def main():
    auth_file = "auth.json"
    url = "https://labs.google/fx/tools/flow"
    
    print("Launching headful browser to debug login...")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            context = browser.new_context(storage_state=auth_file)
            page = context.new_page()
            print(f"Navigating to {url}...")
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            print("Waiting 10 seconds for redirects to settle...")
            page.wait_for_timeout(10000)
            
            print(f"Current URL: {page.url}")
            print(f"Page Title: {page.title()}")
            
            # Save a screenshot to help debug
            screenshot_path = "scratch_login_debug.png"
            page.screenshot(path=screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")
            
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()
