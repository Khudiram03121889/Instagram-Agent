import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force stdout to use backslashreplace for characters it can't print
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
        
        # Let's wait for prompt box to be visible
        prompt_selectors = [
            "div[contenteditable='true'][data-slate-editor='true']",
            "div[contenteditable='true']",
            "textarea[placeholder*='create']"
        ]
        
        editor_loaded = False
        for sel in prompt_selectors:
            try:
                print(f"[DEBUG] Waiting for selector '{sel}'...")
                page.wait_for_selector(sel, timeout=30000)
                print(f"[DEBUG] Editor loaded! Found selector: {sel}")
                editor_loaded = True
                break
            except Exception as e:
                print(f"[DEBUG] Selector '{sel}' not found or timed out: {e}")
                
        if not editor_loaded:
            print("[DEBUG] Failed to load editor in 30 seconds.")
            page.screenshot(path="scratch/editor_load_failed.png")
            browser.close()
            return
            
        page.wait_for_timeout(2000) # Give extra time for model pills to render
        page.screenshot(path="scratch/editor_loaded.png")
        print("[DEBUG] Saved scratch/editor_loaded.png")
        
        # Try to find the settings/model pill button
        pill_btn = None
        selectors = [
            "button[id*='radix-']:has-text('x2')",
            "button[id*='radix-']:has-text('Banana')",
            "button.sc-93abd9dc-1",
            "button:has-text('x2')",
            "button[aria-haspopup='menu']"
        ]
        
        for sel in selectors:
            loc = page.locator(sel)
            count = loc.count()
            print(f"[DEBUG] Selector '{sel}' matched {count} elements.")
            for i in range(count):
                el = loc.nth(i)
                if el.is_visible():
                    txt = el.inner_text().strip().replace('\n', ' ')
                    print(f"   - Match {i}: text='{txt}', class='{el.get_attribute('class')}', id='{el.get_attribute('id')}'")
                    if "x2" in txt or "Banana" in txt or "crop" in txt:
                        pill_btn = el
                        print(f"   => Selected Match {i} as the Settings Pill!")
            if pill_btn:
                break
                
        if pill_btn:
            print("[DEBUG] Clicking Settings Pill...")
            pill_btn.click()
            page.wait_for_timeout(3000)
            
            page.screenshot(path="scratch/settings_open.png")
            print("[DEBUG] Saved scratch/settings_open.png")
            
            # Print all elements inside any open popovers or dialogs
            portals = page.locator('div[data-radix-portal], div[role="dialog"], [role="menu"]').all()
            print(f"[DEBUG] Found {len(portals)} portals/dialogs/menus open:")
            for idx, portal in enumerate(portals):
                print(f"--- Portal {idx} ---")
                print(f"Text: '{portal.inner_text().strip().replace(chr(10), ' | ')[:500]}'")
                
                # Print all buttons in this portal
                btns = portal.locator('button, [role="tab"], [role="menuitem"], [role="option"]').all()
                for b_idx, btn in enumerate(btns):
                    tag = btn.evaluate("el => el.tagName").lower()
                    role = btn.get_attribute("role") or ""
                    text = btn.inner_text().strip().replace('\n', ' ')
                    selected = btn.get_attribute("aria-selected") or btn.get_attribute("data-state") or ""
                    print(f"  Btn {b_idx}: <{tag}> role='{role}' text='{text}' state='{selected}' class='{btn.get_attribute('class')}'")
        else:
            print("[DEBUG] Could not find the settings pill button!")
            
        browser.close()

if __name__ == "__main__":
    main()
