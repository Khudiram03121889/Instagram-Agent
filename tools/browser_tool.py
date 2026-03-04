import os
import time
import json
from typing import Type, List, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from playwright.sync_api import sync_playwright

class VideoGenInput(BaseModel):
    """Input schema for VideoGenerationTool."""
    prompts: Optional[List[str]] = Field(None, description="List of visual prompts to generate videos for. (Optional if json_content is provided)")
    json_content: Optional[str] = Field(None, description="The raw JSON string output from the Prompt Engineer. The tool will extract 'visual' fields automatically.")
    url: str = Field(..., description="The URL of Google Flow (https://labs.google/fx/tools/flow).")
    project_name: str = Field(None, description="Optional project name.")


def _build_cinematic_prompt(item: dict, clip_number: int = 1) -> str:
    """
    Builds a rich, Veo-native cinematic prompt adapted for 8-second Science Documentary clips.

    Conscious Adaptation for Science (5-Clip System):
    - Clip 1 (Hook)      => HYPNOTIC ENTRY (Slow Push In). Viewer enters the microscopic/cosmic world.
    - Clip 2 (Curiosity)  => LEAN-IN (Gentle Dolly Forward). Visual invitation to look closer.
    - Clip 3 (Reason)     => SCIENTIFIC OBSERVATION (Lateral Tracking). Watching the process unfold.
    - Clip 4 (Meaning)    => CONTEXT REVEAL (Slow Pull Back). Seeing the bigger picture/system.
    - Clip 5 (Anchor)     => HUMAN CONNECTION (Static/Breathe). Grounding the concept in reality.

    AUDIO-FIRST DESIGN (V3):
    Veo 3.1 prioritizes elements that appear early in the prompt. The narrator
    dialogue MUST be the first element to ensure audio generation is triggered.

    Key rules for reliable audio on Veo 3.1 Fast:
    1. Dialogue first, using colon format (not quotation marks)
    2. Total prompt under ~120 words
    3. Never say "No music" or "No audio" — use positive framing only
    4. Explicit ambient SFX request reinforces audio generation
    """
    # Extract fields
    video_style = item.get("video_style", "cinematic documentary").strip()
    visual      = item.get("visual", "").strip()
    voice_text  = item.get("voice_text", "").strip()
    audio       = item.get("background_audio", {})
    audio_type  = audio.get("type")
    if not audio_type:
        raise ValueError(
            f"CRITICAL: background_audio.type missing from clip {clip_number} JSON. "
            "The topic_classifier assigns unique audio per category — this field is required."
        )
    
    # Extract topic_classifier fields
    narrator_mood = item.get("narrator_delivery", "Calm, documentary")
    sfx_layers    = item.get("sfx_layers", audio_type)

    # -------------------------------------------------------------------
    # 1. CAMERA — concise, clip-role-aware
    # -------------------------------------------------------------------
    if clip_number == 1:
        # HOOK: Enter the world. Not just "Crane Down" (too fast).
        # "Slow Push In" creates immersion/awe.
        camera = (
            "SLOW PUSH IN (forward dolly), starting wide and drifting closer to the subject. "
            "Wide angle 24mm lens establishing scale. "
            "Slow motion 60fps for hypnotic, grand feeling."
        )
    elif clip_number == 2:
        # CURIOSITY: Lean in. Visual invitation to look closer.
        # Gentle dolly forward, tighter than the hook but not yet process-level.
        camera = (
            "GENTLE DOLLY FORWARD (slow lean-in), moving from medium-wide to medium shot. "
            "35mm lens, slightly tighter than the establishing shot. "
            "Standard 24fps motion, creating a visual invitation to examine closer."
        )
    elif clip_number == 3:
        # REASON: Observe the mechanism. "Lateral Tracking" sounds like a lab observation.
        camera = (
            "LATERAL TRACKING SHOT moving parallel to the process. "
            "Medium 35mm lens. "
            "Standard 24fps motion, observing the reaction/movement clearly."
        )
    elif clip_number == 4:
        # MEANING: The realization. "Slow Pull Back" reveals the system/context.
        camera = (
            "SLOW PULL BACK (backward dolly) to reveal the larger system/context. "
            "Macro 50mm lens transitioning to wide. "
            "Slow motion 60fps emphasizing the realization/scale shift."
        )
    else: # Clip 5
        # ANCHOR: The human moment.
        camera = (
            "STATIC PORTRAIT with SUBTLE BREATHE (very gentle handheld movement). "
            "Eye-level medium shot. 85mm portrait lens. "
            "Natural 24fps motion, intimate and grounded."
        )

    # -------------------------------------------------------------------
    # 2. LIGHTING — concise
    # -------------------------------------------------------------------
    lighting_map = {
        1: "Cinematic volumetric lighting with deep shadows and clear focal point.",
        2: "Soft directional light, creating gentle depth and inviting the eye forward.",
        3: "Soft, even studio lighting (natural or scientific) to show detail.",
        4: "Dramatic contrast usage to emphasize the subject against the void/background.",
        5: "Warm, natural window light or golden hour glow. Human and real.",
    }
    lighting = lighting_map.get(clip_number, lighting_map[1])

    # -------------------------------------------------------------------
    # FINAL PROMPT ASSEMBLY — DIALOGUE FIRST
    # -------------------------------------------------------------------
    # Veo best practice: use "A narrator says:" with colon (not quotes)
    # to trigger voice generation instead of subtitle rendering.
    # Keep total prompt concise (~100-120 words).
    prompt = (
        f"A calm male narrator says: {voice_text}. "
        f"{camera}. "
        f"{visual}. {lighting}. "
        f"{video_style}. "
        f"Ambient sound: {sfx_layers}. "
        f"Portrait 9:16, clean composition."
    )

    return prompt





class VideoGenerationTool(BaseTool):
    name: str = "Video Generation Browser Tool"
    description: str = (
        "Uses Playwright to generate videos on Google Flow using a saved authentication state. "
        "Requires 'auth.json' to be present (generated by setup_auth.py). "
        "You can provide either a list of 'prompts' OR the raw 'json_content' from the Prompt Engineer."
    )
    args_schema: Type[BaseModel] = VideoGenInput

    def _run(self, url: str, prompts: List[str] = None, json_content: str = None, project_name: str = None) -> str:
        auth_file = "auth.json"
        
        # --- Logic to extract prompts from JSON if provided ---
        final_prompts = []
        if json_content:
            try:
                required_keys = [
                    "orientation",
                    "aspect_ratio",
                    "video_style",
                    "voice",
                    "background_audio",
                    "voice_text",
                    "visual",
                ]
                content = json_content.strip()
                # Attempt to find the first '[' and last ']' to extract the list
                start_index = content.find('[')
                end_index = content.rfind(']')
                
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    potential_json = content[start_index:end_index+1]
                    data = json.loads(potential_json)
                else:
                    # Fallback: try parsing the whole string if it's not a list wrapper (e.g. single dict)
                    # or if brackets weren't found (though expected for a list)
                    data = json.loads(content.replace("```json", "").replace("```", ""))

                # Per-clip word limits aligned with cognitive importance:
                # Hook & Anchor get more breathing room, transition clips are tighter.
                CLIP_WORD_LIMITS = {1: 14, 2: 13, 3: 13, 4: 15, 5: 14}

                def _truncate_at_sentence_boundary(text: str, max_words: int) -> str:
                    """Truncate at the last sentence-ending punctuation before max_words."""
                    words = text.split()
                    if len(words) <= max_words:
                        return text
                    candidate = " ".join(words[:max_words])
                    # Find last sentence boundary (.?!) in the candidate
                    last_boundary = -1
                    for punct in ['. ', '? ', '! ']:
                        idx = candidate.rfind(punct)
                        if idx > last_boundary:
                            last_boundary = idx + 1  # include the punctuation
                    if last_boundary > len(candidate) // 3:  # only use if not too early
                        return candidate[:last_boundary].strip()
                    # Fallback: trim to max_words - 1 for breathing room
                    return " ".join(words[:max_words - 1])

                def normalize_item(item: dict, clip_number: int = 1) -> dict:
                    missing = [k for k in required_keys if k not in item]
                    if missing:
                        raise ValueError(f"Missing required keys: {', '.join(missing)}")
                    
                    # Hardcore validation: Ensure voice_text is present and not empty
                    if not item.get("voice_text") or not str(item["voice_text"]).strip():
                        raise ValueError("CRITICAL: 'voice_text' is empty. Audio generation requires text.")

                    # Per-clip word count guardrail
                    voice_text_str = str(item["voice_text"]).strip()
                    word_count = len(voice_text_str.split())
                    max_words = CLIP_WORD_LIMITS.get(clip_number, 13)
                    if word_count > max_words:
                        trimmed = _truncate_at_sentence_boundary(voice_text_str, max_words)
                        print(f"   ⚠️ Clip {clip_number} voice_text has {word_count} words "
                              f"(max {max_words}). Trimmed: '{trimmed[:50]}...'")
                        item["voice_text"] = trimmed

                    # Preserve a clear, consistent key order
                    # PUT VOICE_TEXT FIRST to emphasize it to the model
                    normalized = {
                        "voice_text": str(item["voice_text"]).strip(),
                        "voice": item["voice"],
                        "background_audio": item["background_audio"],
                        "visual": item["visual"],
                        # Technical settings after content
                        "orientation": item["orientation"],
                        "aspect_ratio": item["aspect_ratio"],
                        "video_style": item["video_style"],
                    }
                    if "clip_label" in item:
                        normalized["clip_label"] = item["clip_label"]
                    if "clip" in item:
                        normalized["clip"] = item["clip"]
                    
                    return normalized

                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            clip_index = len(final_prompts) + 1  # 1-based position so far
                            normalized_item = normalize_item(item, clip_index)
                            
                            # CONVERT TO CINEMATIC NATURAL LANGUAGE PROMPT
                            nl_prompt = _build_cinematic_prompt(normalized_item, clip_index)
                            
                            print(f"   📝 Cinematic Prompt [{clip_index}]: {nl_prompt[:70]}...")
                            final_prompts.append(nl_prompt)
                            
                elif isinstance(data, dict):
                    # Single clip object
                    normalized_item = normalize_item(data, 1)
                    nl_prompt = _build_cinematic_prompt(normalized_item, 1)
                    final_prompts.append(nl_prompt)
                
                print(f"✅ Extracted {len(final_prompts)} prompts from JSON content.")
            except Exception as e:
                print(f"⚠️ Warning: Could not parse 'json_content' automatically: {e}. Trying raw prompts if available.")
        
        # Merge with direct prompts if any (though usually it's one or the other)
        if prompts:
            final_prompts.extend(prompts)
            
        if not final_prompts:
            return "❌ Error: No prompts found. Please provide either 'prompts' list or valid 'json_content' containing 'visual' fields."

        if not os.path.exists(auth_file):
            return "❌ Error: 'auth.json' not found. Please run 'python setup_auth.py' first to login."

        results = []
        
        print(f"\n{'='*60}")
        print("🎬 VIDEO GENERATION TOOL (PLAYWRIGHT)")
        print(f"{'='*60}")
        
        with sync_playwright() as p:
            # Launch browser with the saved storage state
            print("🚀 Launching Chrome...")
            browser = p.chromium.launch(
                headless=False, 
                channel="chrome",
                args=["--start-maximized", "--disable-blink-features=AutomationControlled"]
            ) 
            
            # Create a context with the saved auth state
            try:
                context = browser.new_context(
                    storage_state=auth_file, 
                    viewport=None # Important for maximized window
                )
            except Exception as e:
                return f"❌ Error loading auth.json: {e}. Try running setup_auth.py again."

            page = context.new_page()
            
            try:
                print(f"🌐 Navigating to {url}...")
                page.goto(url)
                page.wait_for_load_state("networkidle")
                
                # Check if we are actually logged in (simple check)
                if "sign in" in page.title().lower():
                     return "❌ Error: It seems you are not logged in. Please run 'python setup_auth.py' again."

                # --- STEP 1: Click "New Project" ---
                print("🆕 Looking for '+ New project' button...")
                try:
                    # Using the exact class string from your screenshot
                    new_project_btn = page.locator("button.fXsrxE, button.sc-a38764c7-0, button:has-text('New project')")
                    new_project_btn.first.click(timeout=10000)
                    print("✅ Clicked 'New project'")
                    time.sleep(3) 
                except Exception as e:
                    print(f"⚠️ Could not find 'New project' button: {e}")
                    # Last resort: click center of bottom area where the button usually is
                    page.mouse.click(640, 650) 

                # --- STEP 1.5 & 1.6: Verify Mode and Settings via Settings Menu ---
                print("\n⚙️ Verifying generation settings...")
                try:
                    # Look for the settings pill in the prompt bar (e.g. Video | x2)
                    settings_btn = page.locator("button[aria-haspopup='menu']").last
                    try:
                        settings_btn.wait_for(state="visible", timeout=8000)
                    except:
                        # Fallback if there's only one or different selectors
                        settings_btn = page.locator("button[id*='radix']").filter(has_text="x2").first
                        settings_btn.wait_for(state="visible", timeout=8000)

                    settings_state = settings_btn.get_attribute("data-state")
                    if settings_state != "open":
                        settings_btn.click()
                        print("   📋 Opened settings menu")
                        time.sleep(2)
                    else:
                        print("   📋 Settings menu already open")

                    # --- Check Video Tab is Selected ---
                    video_tab = page.locator("button[role='tab']:has-text('Video'), button[id$='-trigger-VIDEO']").first
                    if video_tab.is_visible():
                        is_selected = video_tab.get_attribute("aria-selected") == "true" or video_tab.get_attribute("data-state") == "active"
                        if not is_selected:
                            print("   ⚠️ Switching to Video tab...")
                            video_tab.click()
                            time.sleep(1)
                        else:
                            print("   ✅ Video tab selected")
                            
                    # --- Set Portrait (all Instagram reels are portrait 9:16) ---
                    orientation = "portrait"
                    orientation_btn = page.locator("button[role='tab'][id*='-trigger-PORTRAIT'], button[role='tab']:has-text('Portrait')").first
                        
                    if orientation_btn.is_visible():
                        is_active = orientation_btn.get_attribute("aria-selected") == "true" or orientation_btn.get_attribute("data-state") == "active" or orientation_btn.get_attribute("aria-pressed") == "true"
                        if not is_active:
                            orientation_btn.click()
                            time.sleep(1)
                            print(f"   ✅ Set to '{orientation}'")
                        else:
                            print(f"   ✅ '{orientation}' mode already selected")

                    # --- Check Duration (x2) ---
                    x2_btn = page.locator("button[role='tab']:has-text('x2')").first
                    if x2_btn.is_visible():
                        is_active = x2_btn.get_attribute("aria-selected") == "true" or x2_btn.get_attribute("data-state") == "active"
                        if not is_active:
                            x2_btn.click()
                            time.sleep(1)
                            print("   ✅ Set to x2")
                        else:
                            print("   ✅ x2 already selected")
                            
                    # --- Check Model (Veo 3.1 - Fast) ---
                    try:
                        model_dropdown = page.locator("button[role='combobox']:has-text('Veo'), select:has-text('Veo')").first
                        if model_dropdown.is_visible():
                            model_val = model_dropdown.inner_text().strip()
                            if "3.1 - Fast" not in model_val:
                                model_dropdown.click()
                                time.sleep(1)
                                page.locator("[role='option']:has-text('Veo 3.1 - Fast')").first.click()
                                print("   ✅ Set model to Veo 3.1 - Fast")
                    except Exception as e:
                        print(f"   ⚠️ Could not set model: {e}")

                    # Close settings panel (click outside or press Esc)
                    page.keyboard.press("Escape")
                    time.sleep(1)

                except Exception as e:
                    print(f"   ⚠️ Could not open/verify settings menu: {e}")
                    print("   Proceeding with prompt entry anyway...")

                # --- STEP 2: Process Prompts ---
                for i, prompt in enumerate(final_prompts):
                    clip_num = i + 1
                    print(f"\n📌 Processing Clip {clip_num}/{len(final_prompts)}: '{prompt[:30]}...'")
                    
                    # Locate Prompt Box
                    print("   🔍 Finding prompt box...")
                    # New UI uses a contenteditable div with data-slate-editor
                    prompt_box = page.locator("div[contenteditable='true'][data-slate-editor='true']").first
                    
                    if not prompt_box.is_visible():
                        # Fallback to old selectors just in case
                        prompt_box = page.locator("textarea[placeholder*='Generate'], textarea.gEBbLp, #PINHOLE_TEXT_AREA_ELEMENT_ID").first
                    
                    prompt_box.fill(prompt)
                    print("   ✍️ Prompt entered")
                    time.sleep(1)

                    # Locate Generate Button
                    print("   🚀 Clicking Generate...")
                    # The arrow button is typically right next to the input
                    generate_btn = page.locator("button:has(i:has-text('arrow_forward')), button.gdArnN, button[aria-label*='Send']").first
                    
                    # --- STEP 3: Generate with Retry Logic ---
                    MAX_RETRIES = 2
                    clip_succeeded = False
                    
                    for attempt in range(MAX_RETRIES):
                        attempt_label = f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        try:
                            if attempt > 0:
                                print(f"   🔄 Retrying clip {clip_num} {attempt_label}...")
                                # Re-enter prompt for retry
                                prompt_box.fill(prompt)
                                time.sleep(1)
                            
                            generate_btn.click()
                            print(f"   ✅ Generate clicked! {attempt_label}")
                            
                            print(f"   ⏳ Waiting for generation (timeout 3 mins) {attempt_label}...")
                            page.wait_for_timeout(10000)  # Initial wait
                            page.locator(".loading-spinner, .progress-bar").wait_for(state="hidden", timeout=180000)
                            
                            results.append(f"Clip {clip_num}: ✅ Generated")
                            clip_succeeded = True
                            break  # Success — exit retry loop
                            
                        except Exception as e:
                            print(f"   ⚠️ {attempt_label} failed: {e}")
                            if attempt < MAX_RETRIES - 1:
                                print(f"   ⏳ Waiting 5 seconds before retry...")
                                time.sleep(5)
                    
                    if not clip_succeeded:
                        results.append(f"Clip {clip_num}: ❌ FAILED after {MAX_RETRIES} attempts")
                        print(f"   ❌ Clip {clip_num} permanently failed after {MAX_RETRIES} attempts")
                    
                    time.sleep(2)
                
                # --- STEP 4: Rename Project (Optional) ---
                if project_name:
                    print(f"\n🏷️ Renaming project to '{project_name}'...")
                    try:
                        title_input = page.locator("input[aria-label='Editable text'], input[value*='Mar'], [class*='fpqFyW']").first
                        if title_input.is_visible():
                            title_input.click()
                            page.keyboard.press("Control+A")
                            page.keyboard.press("Backspace")
                            title_input.fill(project_name)
                            page.keyboard.press("Enter")
                            print("   ✅ Project renamed")
                        else:
                            print("   ⚠️ Could not locate project title input")
                    except Exception as e:
                        print(f"   ⚠️ Could not rename project: {e}")

            except Exception as e:
                msg = f"❌ Error during automation: {e}"
                print(msg)

                results.append(msg)
            
            print(f"\n{'='*60}")
            print("🏁 FINISHED. Browser will remain open for 30 seconds for you to review/download.")
            time.sleep(30)
            
            return "\n".join(results)
