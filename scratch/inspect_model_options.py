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
        
        # Model selector dropdown trigger is Btn 11 which has text like "\U0001f34c Nano Banana 2 arrow_drop_down" or similar.
        # Let's locate it:
        model_dropdown = page.locator("button:has-text('Banana'), button:has-text('Veo'), button:has-text('Omni'), button:has-text('arrow_drop_down')").last
        if not model_dropdown.is_visible():
            print("[DEBUG] Model dropdown button not visible!")
            browser.close()
            return
            
        print(f"[DEBUG] Clicking model dropdown (Text: '{model_dropdown.inner_text().strip().replace(chr(10), ' ')}')...")
        model_dropdown.click()
        page.wait_for_timeout(2000)
        
        page.screenshot(path="scratch/model_dropdown_open.png")
        print("[DEBUG] Saved scratch/model_dropdown_open.png")
        
        # List all portal/menu/dialog/popover elements
        portals = page.locator('div[data-radix-portal], [role="menu"], [role="listbox"]').all()
        print(f"[DEBUG] Found {len(portals)} portals/menus open:")
        for idx, portal in enumerate(portals):
            print(f"--- Portal/Menu {idx} ---")
            print(f"Text: '{portal.inner_text().strip().replace(chr(10), ' | ')[:1000]}'")
            
            # Print all buttons/options in this menu
            items = portal.locator('button, [role="menuitem"], [role="menuitemradio"], [role="option"], span').all()
            for i_idx, item in enumerate(items):
                try:
                    if item.is_visible():
                        tag = item.evaluate("el => el.tagName").lower()
                        txt = item.inner_text().strip().replace('\n', ' ')
                        role = item.get_attribute("role") or ""
                        print(f"  Item {i_idx}: <{tag}> role='{role}' text='{txt}' class='{item.get_attribute('class')}'")
                except Exception:
                    pass
                    
        browser.close()

if __name__ == "__main__":
    main()
