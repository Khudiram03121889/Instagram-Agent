import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.browser_tool

# Monkeypatch SETTINGS_MENU_SELECTORS to target the settings pill button at the bottom of the editor
tools.browser_tool.SETTINGS_MENU_SELECTORS = [
    "button[aria-haspopup='menu']:has-text('x2')",
    "button[aria-haspopup='menu']:has-text('x1')",
    "button[aria-haspopup='menu']:has-text('x3')",
    "button[aria-haspopup='menu']:has-text('x4')",
    "button[aria-haspopup='menu']:has-text('Banana')",
    "button[aria-haspopup='menu']:has-text('Veo')",
    "button[aria-haspopup='menu']:has-text('Omni')",
    "button[aria-haspopup='menu']:has-text('Flash')",
    "button[aria-haspopup='menu']",
    "button[id*='radix']",
]

# Monkeypatch MODEL_CONTROL_SELECTORS to target only the trigger button structurally inside the settings menu
tools.browser_tool.MODEL_CONTROL_SELECTORS = [
    "[role='menu'] button[aria-haspopup='menu']",
    "[role='menu'] button[role='combobox']",
    "[role='menu'] button:has-text('arrow_drop_down')",
    "button[role='combobox']:has-text('Veo')",
    "button[aria-haspopup='menu']:has-text('Omni')",
    "button:has-text('Banana')",
]

# Monkeypatch MODEL_OPTION_SELECTORS to target Radix dropdown options first and prevent background clicks
tools.browser_tool.MODEL_OPTION_SELECTORS = [
    "[role='menuitem']:has-text('Nano Banana 2')",
    "[role='menuitem']:has-text('Nano Banana Pro')",
    "[role='menuitem']:has-text('Omni Flash')",
    "[role='menuitemradio']:has-text('Nano Banana 2')",
    "[role='menuitemradio']:has-text('Nano Banana Pro')",
    "[role='menuitemradio']:has-text('Omni Flash')",
    "[role='option']:has-text('Nano Banana 2')",
    "[role='option']:has-text('Nano Banana Pro')",
    "[role='option']:has-text('Omni Flash')",
    "button:has-text('Nano Banana 2')",
    "button:has-text('Nano Banana Pro')",
    "button:has-text('Omni Flash')",
]

# Module-level globals to track prompts and clip data
prompts_data = []
final_prompts_list = []

def custom_read_locator_text(locator) -> str:
    try:
        value = locator.evaluate(
            """(el) => {
                if ('value' in el && typeof el.value === 'string') {
                    return el.value;
                }
                return (el.innerText || el.textContent || '').trim();
            }"""
        )
        return tools.browser_tool._normalize_space(value)
    except Exception as e:
        print(f"   [DEBUG] _read_locator_text error: {e}")
        return ""

tools.browser_tool._read_locator_text = custom_read_locator_text

def _ensure_duration_selected(page, target_seconds: int) -> None:
    settings_btn = None
    for selector in tools.browser_tool.SETTINGS_MENU_SELECTORS:
        candidate = page.locator(selector).last
        try:
            if candidate.count() > 0 and candidate.is_visible():
                settings_btn = candidate
                break
        except Exception:
            continue
    if settings_btn is None:
        print("[WARNING] Could not locate Flow settings menu to change duration.")
        return

    # Check if the settings menu is open. If not, open it using robust aria-expanded check and click
    if settings_btn.get_attribute("aria-expanded") != "true":
        settings_btn.click()
        page.wait_for_timeout(1000)

    # Build the duration tab string
    duration_str = f"{target_seconds}s"
    
    # Try different selectors to click the specific duration option
    selectors = [
        f"[role='tab']:has-text('{duration_str}')",
        f"button[role='tab']:has-text('{duration_str}')",
        f"button:has-text('{duration_str}')",
        f"span:has-text('{duration_str}')",
    ]
    
    selected = False
    for selector in selectors:
        tab = page.locator(selector).first
        try:
            if tab.count() > 0 and tab.is_visible():
                if tab.get_attribute("aria-selected") == "true":
                    print(f"   [DEBUG] Duration '{duration_str}' is already selected.")
                    selected = True
                    break
                else:
                    tab.click()
                    print(f"   [DEBUG] Selected duration '{duration_str}' via selector: '{selector}'")
                    page.wait_for_timeout(800)
                    selected = True
                    break
        except Exception as e:
            continue
            
    if not selected:
        print(f"   [WARNING] Could not select duration '{duration_str}' in settings menu.")

    # Close settings menu
    try:
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
    except Exception:
        pass

def custom_verify_flow_settings(page, target_seconds: int = None) -> None:
    global prompts_data
    if target_seconds is None:
        if len(prompts_data) > 0:
            target_seconds = prompts_data[0]["duration_seconds"]
        else:
            target_seconds = 8  # Fallback default

    page.wait_for_timeout(2000) # Give extra time for settings pill to render
    settings_btn = None
    last_error = None
    for selector in tools.browser_tool.SETTINGS_MENU_SELECTORS:
        candidate = page.locator(selector).first
        try:
            if candidate.count() > 0 and candidate.is_visible():
                settings_btn = candidate
                break
        except Exception as exc:
            last_error = exc
            
    if settings_btn is None:
        for selector in tools.browser_tool.SETTINGS_MENU_SELECTORS:
            candidate = page.locator(selector).last
            try:
                if candidate.count() > 0 and candidate.is_visible():
                    settings_btn = candidate
                    break
            except Exception as exc:
                last_error = exc

    if settings_btn is None:
        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(f"Could not locate Flow settings menu.{detail}")

    if settings_btn.get_attribute("aria-expanded") != "true":
        settings_btn.click()
        page.wait_for_timeout(1200)

    try:
        video_tab = page.locator(
            "button[role='tab']:has-text('Video'), button[id$='-trigger-VIDEO']"
        ).first
        portrait_tab = page.locator(
            "button[role='tab'][id*='-trigger-PORTRAIT'], button[role='tab']:has-text('Portrait')"
        ).first
        x2_tab = page.locator("button[role='tab']:has-text('x2')").first

        tools.browser_tool._ensure_selected(video_tab, "Video mode", page=page)
        tools.browser_tool._ensure_selected(portrait_tab, "Portrait mode", page=page)
        tools.browser_tool._ensure_selected(x2_tab, "x2 duration", page=page)

        # Select target seconds (e.g. 4s, 6s, 8s, 10s)
        duration_str = f"{target_seconds}s"
        duration_selectors = [
            f"[role='menu'] button[role='tab']:has-text('{duration_str}')",
            f"[role='menu'] [role='tab']:has-text('{duration_str}')",
            f"[role='menu'] button:has-text('{duration_str}')",
            f"[role='menu'] span:has-text('{duration_str}')",
            f"[role='tab']:has-text('{duration_str}')",
            f"button[role='tab']:has-text('{duration_str}')",
        ]
        
        duration_tab = None
        for selector in duration_selectors:
            candidate = page.locator(selector).first
            try:
                if candidate.count() > 0 and candidate.is_visible():
                    duration_tab = candidate
                    break
            except Exception:
                continue
                
        if duration_tab is None:
            print(f"   [WARNING] Could not locate duration option for {duration_str}.")
        else:
            tools.browser_tool._ensure_selected(duration_tab, f"{duration_str} duration", page=page)

        # Locate the model selector trigger button structurally
        model_control = None
        for selector in tools.browser_tool.MODEL_CONTROL_SELECTORS:
            candidate = page.locator(selector).first
            try:
                if candidate.count() > 0 and candidate.is_visible():
                    model_control = candidate
                    break
            except Exception:
                continue
        if model_control is None:
            raise RuntimeError("Could not locate the Flow model selector.")

        try:
            outer_html = model_control.evaluate("el => el.outerHTML")
            print(f"[DEBUG] Matched model selector outer HTML: '{outer_html}'")
            inner_text = model_control.evaluate("el => el.innerText")
            text_content = model_control.evaluate("el => el.textContent")
            print(f"[DEBUG] el.innerText: '{inner_text}'")
            print(f"[DEBUG] el.textContent: '{text_content}'")
        except Exception as e:
            print(f"[DEBUG] Failed to evaluate model selector HTML: {e}")

        # Evaluate text directly using javascript instead of read_locator_text (bypassing namespace issues)
        model_text = tools.browser_tool._normalize_space(model_control.evaluate("el => el.innerText || el.textContent"))
        print(f"[DEBUG] Current model control button text is: '{model_text}'")
        if not any(m in model_text for m in ["Nano Banana 2", "Nano Banana Pro", "Omni Flash", "Veo 3.1 - Fast"]):
            model_control.click()
            print("[DEBUG] Clicked model selector dropdown. Waiting for options to render...")
            page.wait_for_timeout(1500)
            
            # Debug log all text nodes on the page that could be option elements
            try:
                all_texts = page.evaluate("() => Array.from(document.querySelectorAll('*')).map(el => el.innerText || el.value).filter(Boolean)")
                seen = set()
                print("[DEBUG] Model list options / text nodes currently present on the page:")
                for text in all_texts:
                    text_clean = text.strip()
                    if text_clean and text_clean not in seen and any(w in text_clean.lower() for w in ["omni", "flash", "veo", "banana", "nano", "model"]):
                        print(f"   - '{text_clean}'")
                        seen.add(text_clean)
            except Exception as e:
                print(f"   [DEBUG] Failed to list text nodes: {e}")

            selected = False
            for selector in tools.browser_tool.MODEL_OPTION_SELECTORS:
                option = page.locator(selector).first
                try:
                    if option.count() > 0 and option.is_visible():
                        print(f"[DEBUG] Found model option matching selector '{selector}': text='{option.inner_text()}'")
                        option.click()
                        selected = True
                        break
                except Exception as e:
                    print(f"   [DEBUG] Error checking selector '{selector}': {e}")
                    continue
            if not selected:
                # Capture dropdown screenshot and HTML before Escape closes it
                os.makedirs("outputs/Why do we close our eyes when we sneeze", exist_ok=True)
                try:
                    page.screenshot(path="outputs/Why do we close our eyes when we sneeze/flow_dropdown_failed.png", timeout=5000)
                except Exception as e:
                    print(f"[WARNING] Safe screenshot failed: {e}")
                with open("outputs/Why do we close our eyes when we sneeze/flow_dropdown_failed.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                raise RuntimeError("Could not locate the model option.")
            
            page.wait_for_timeout(1500)
            
            # Re-locate the settings button first to see if the panel closed
            settings_btn = None
            for selector in tools.browser_tool.SETTINGS_MENU_SELECTORS:
                candidate = page.locator(selector).last
                try:
                    if candidate.count() > 0 and candidate.is_visible():
                        settings_btn = candidate
                        break
                except Exception:
                    continue
            if settings_btn is None:
                raise RuntimeError("Could not locate Flow settings menu during verification.")

            # If the panel closed, re-open it to verify the settings
            if settings_btn.get_attribute("aria-expanded") != "true":
                print("[DEBUG] Settings panel closed after option click. Re-opening it to verify...")
                settings_btn.click()
                page.wait_for_timeout(1500)

            # Re-locate the model button structurally (resolves stale/detached element error)
            model_control = None
            for selector in tools.browser_tool.MODEL_CONTROL_SELECTORS:
                candidate = page.locator(selector).first
                try:
                    if candidate.count() > 0 and candidate.is_visible():
                        model_control = candidate
                        break
                except Exception:
                    continue
            if model_control is None:
                raise RuntimeError("Could not re-locate the Flow model selector after selection.")

            try:
                outer_html = model_control.evaluate("el => el.outerHTML")
                print(f"[DEBUG] Re-located model selector outer HTML: '{outer_html}'")
                inner_text = model_control.evaluate("el => el.innerText")
                text_content = model_control.evaluate("el => el.textContent")
                print(f"[DEBUG] el.innerText: '{inner_text}'")
                print(f"[DEBUG] el.textContent: '{text_content}'")
            except Exception as e:
                print(f"[DEBUG] Failed to evaluate re-located model selector HTML: {e}")

            # Evaluate text directly using javascript instead of read_locator_text (bypassing namespace issues)
            model_text = tools.browser_tool._normalize_space(model_control.evaluate("el => el.innerText || el.textContent"))
            print(f"[DEBUG] Re-located model control button text is: '{model_text}'")
            if not any(m in model_text for m in ["Nano Banana 2", "Nano Banana Pro", "Omni Flash", "Veo 3.1 - Fast"]):
                # Capture mismatch screenshot and HTML
                os.makedirs("outputs/Why do we close our eyes when we sneeze", exist_ok=True)
                try:
                    page.screenshot(path="outputs/Why do we close our eyes when we sneeze/flow_dropdown_failed.png", timeout=5000)
                except Exception as e:
                    print(f"[WARNING] Safe screenshot failed: {e}")
                with open("outputs/Why do we close our eyes when we sneeze/flow_dropdown_failed.html", "w", encoding="utf-8") as f:
                    f.write(page.content())
                raise RuntimeError("Model selector did not confirm selected model.")
    finally:
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        except Exception:
            pass

tools.browser_tool._verify_flow_settings = custom_verify_flow_settings

original_build_cinematic_prompt = tools.browser_tool._build_cinematic_prompt

def custom_build_cinematic_prompt(item, clip_number=1):
    prompt = original_build_cinematic_prompt(item, clip_number)
    final_prompts_list.append(prompt)
    return prompt

tools.browser_tool._build_cinematic_prompt = custom_build_cinematic_prompt

original_enter_prompt = tools.browser_tool._enter_prompt

def custom_enter_prompt(prompt_box, prompt):
    page = prompt_box.page if hasattr(prompt_box, "page") else None
    if page:
        try:
            clip_index = None
            for idx, p in enumerate(final_prompts_list):
                if p.strip() == prompt.strip() or p.strip().startswith(prompt.strip()[:100]):
                    clip_index = idx
                    break
            
            if clip_index is not None and clip_index < len(prompts_data):
                duration = prompts_data[clip_index]["duration_seconds"]
                print(f"[RUNNER] Clip {clip_index + 1} requires {duration}s duration. Ensuring settings are correct...")
                # Call our full verify flow settings function with the specific target duration
                custom_verify_flow_settings(page, target_seconds=duration)
            else:
                print(f"[WARNING] Could not match prompt (length {len(prompt)}) to any clip to verify duration.")
        except Exception as e:
            print(f"[WARNING] Error checking/setting settings: {e}")
            
    original_enter_prompt(prompt_box, prompt)

tools.browser_tool._enter_prompt = custom_enter_prompt

from tools.browser_tool import VideoGenerationTool

def main():
    global prompts_data
    # Force UTF-8 stdout
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    prompts_path = os.path.abspath(
        "outputs/Why do we close our eyes when we sneeze/validated_prompts.json"
    )
    print(f"[RUNNER] Loading validated prompts from: {prompts_path}")
    
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        
    print(f"[RUNNER] Extracted {len(prompts_data)} clips.")
    
    tool = VideoGenerationTool()
    tool.result_as_answer = True
    
    print("[RUNNER] Launching Google Flow browser operator stage...")
    res = tool._run(
        url="https://labs.google/fx/tools/flow",
        json_content=json.dumps(prompts_data),
        project_name="Why do we close our eyes when we sneeze",
        dry_run=False
    )
    print(f"[RUNNER] Result:\n{res}")

if __name__ == "__main__":
    main()
