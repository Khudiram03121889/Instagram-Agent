import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(errors='backslashreplace')
except Exception:
    pass

from playwright.sync_api import sync_playwright

def main():
    with sync_playwright() as p:
        print("[DEBUG] Launching chromium...")
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state='auth.json')
        page = context.new_page()
        
        print("[DEBUG] Navigating to Google Flow...")
        page.goto('https://labs.google/fx/tools/flow', timeout=30000)
        page.wait_for_timeout(3000)
        
        # Click "New project" button if we are not in editor
        try:
            new_btn = page.locator("button:has-text('New project'), button.fXsrxE").first
            if new_btn.is_visible(timeout=5000):
                print("[DEBUG] Clicking New project...")
                new_btn.click()
        except Exception as e:
            print(f"[DEBUG] Error clicking New project: {e}")
            
        print(f"[DEBUG] Current URL: {page.url}")
        
        # Wait for prompt box
        page.wait_for_selector("div[contenteditable='true'][data-slate-editor='true']", timeout=30000)
        page.wait_for_timeout(2000)
        
        # Find settings pill button
        pill_btn = page.locator("button[id*='radix-']:has-text('x2'), button[aria-haspopup='menu']:has-text('x2'), button:has-text('x2')").first
        if not pill_btn.is_visible():
            print("[DEBUG] Settings Pill button not visible!")
            browser.close()
            return
            
        print("[DEBUG] Clicking Settings Pill...")
        pill_btn.click()
        page.wait_for_timeout(2000)
        
        # Let's test the specific portal-prefixed locators!
        print("[DEBUG] Verifying portal locators:")
        
        video_tab = page.locator("div[data-radix-portal] button[role='tab']:has-text('Video')").first
        portrait_tab = page.locator("div[data-radix-portal] button[role='tab']:has-text('9:16'), div[data-radix-portal] button[role='tab']:has-text('Portrait')").first
        x2_tab = page.locator("div[data-radix-portal] button[role='tab']:has-text('x2'), div[data-radix-portal] button[role='tab']:has-text('2X')").first
        
        for name, loc in [("Video Tab", video_tab), ("Portrait Tab", portrait_tab), ("X2 Tab", x2_tab)]:
            count = loc.count()
            print(f"  - {name}: count={count}")
            if count > 0:
                print(f"    text='{loc.inner_text().strip().replace(chr(10), ' ')}', visible={loc.is_visible()}")
                
        browser.close()

if __name__ == "__main__":
    main()
