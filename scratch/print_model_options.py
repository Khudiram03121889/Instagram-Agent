import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from playwright.sync_api import sync_playwright
from tools.browser_tool import NEW_PROJECT_SELECTORS, PROMPT_BOX_SELECTORS

def main():
    with sync_playwright() as p:
        print("[DEBUG] Launching chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state='auth.json')
        page = context.new_page()
        
        print("[DEBUG] Navigating to Google Flow...")
        page.goto('https://labs.google/fx/tools/flow', timeout=30000)
        page.wait_for_timeout(5000)
        
        # Ensure editor open
        editor_ready = False
        for sel in PROMPT_BOX_SELECTORS:
            try:
                locator = page.locator(sel).first
                if locator.count() > 0 and locator.is_visible():
                    print(f"[DEBUG] Editor open! Found: {sel}")
                    editor_ready = True
                    break
            except Exception:
                pass
                
        if not editor_ready:
            try:
                new_project_btn = page.locator(NEW_PROJECT_SELECTORS[0]).first
                if new_project_btn.is_visible(timeout=5000):
                    print("[DEBUG] Clicking New Project...")
                    new_project_btn.click()
                    page.wait_for_timeout(8000)
            except Exception as e:
                print(f"[DEBUG] Error checking new project button: {e}")
                
        # Find settings button
        print("[DEBUG] Locating settings button...")
        settings_btn = None
        selectors = [
            "button.sc-93abd9dc-1",
            "button:has-text('Video')",
            "button:has(i:has-text('settings_2'))",
            "button[aria-haspopup='menu']",
        ]
        for sel in selectors:
            try:
                locator = page.locator(sel).last
                if locator.count() > 0 and locator.is_visible():
                    settings_btn = locator
                    print(f"[DEBUG] Found settings button using: '{sel}'! Text: '{locator.inner_text().strip()}'")
                    break
            except Exception:
                pass
                
        if settings_btn is not None:
            print("[DEBUG] Clicking settings button...")
            settings_btn.click()
            page.wait_for_timeout(3000)
            
            # Print ALL visible buttons on the page now!
            print("[DEBUG] Scanning for all visible buttons on the page...")
            buttons = page.locator('button').all()
            print(f"[DEBUG] Found {len(buttons)} total buttons on the page:")
            for b_idx, btn in enumerate(buttons):
                try:
                    if btn.is_visible():
                        print("  Btn %d: Class='%s', Role='%s', Text='%s', Attrs=%s" % (
                            b_idx,
                            btn.get_attribute('class'),
                            btn.get_attribute('role'),
                            btn.inner_text().strip().replace('\n', ' '),
                            {attr: btn.get_attribute(attr) for attr in ['aria-haspopup', 'aria-expanded', 'aria-selected', 'data-state'] if btn.get_attribute(attr)}
                        ))
                except Exception as e:
                    pass
        else:
            print("[DEBUG] Settings button not visible or not found.")
            
        browser.close()

if __name__ == "__main__":
    main()
