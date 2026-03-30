import os
import time
import json
import re
from datetime import datetime, timezone
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
    dry_run: bool = Field(False, description="Validate prompts locally but skip browser launch and Flow credit usage.")


MIN_CLIP_COUNT = 5
MAX_CLIP_COUNT = 6
MIN_WORD_COUNT = 8
MAX_WORD_COUNT = 42
MAX_SENTENCE_COUNT = 3
EXPECTED_CLIP_ROLE_SEQUENCES = {
    5: ["Hook", "Question", "Mechanism", "Contrast/Payoff", "Personal Takeaway"],
    6: [
        "Hook",
        "Question",
        "Mechanism Part 1",
        "Mechanism Part 2",
        "Contrast/Payoff",
        "Personal Takeaway",
    ],
}
REQUIRED_BACKGROUND_AUDIO_KEYS = (
    "generate_with_video",
    "type",
    "volume",
    "sfx_layers",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "has", "have", "how", "if", "in", "into", "is", "it", "its", "of", "on",
    "or", "so", "that", "the", "their", "them", "there", "these", "this", "to",
    "was", "we", "what", "when", "where", "which", "why", "with", "you", "your"
}
BROWSER_POLL_MS = 500
PROMPT_BOX_SELECTORS = [
    "div[contenteditable='true'][data-slate-editor='true']",
    "div[contenteditable='true'][role='textbox']",
    "div[contenteditable='true']",
    "textarea[placeholder*='Generate']",
    "textarea.gEBbLp",
    "#PINHOLE_TEXT_AREA_ELEMENT_ID",
    "textarea",
]
NEW_PROJECT_SELECTORS = [
    "button.fXsrxE, button.sc-a38764c7-0, button:has-text('New project')",
]
GENERATE_BUTTON_SELECTORS = [
    "button:has(i:has-text('arrow_forward')), button.gdArnN, button[aria-label*='Send']",
]
ACTIVE_GENERATION_SELECTORS = [
    "[role='progressbar']",
    "[aria-label*='Generating']",
    "[aria-label*='Rendering']",
    "[aria-label*='Queued']",
    "[data-testid*='progress']",
    ".loading-spinner",
    ".progress-bar",
]
RESULT_TILE_SELECTORS = [
    "video",
    "figure",
    "[data-testid*='asset']",
    "[data-testid*='result']",
    "[data-testid*='generation']",
]
GENERATION_STATE_KEYWORDS = (
    "generating",
    "rendering",
    "queued",
    "preparing",
    "processing",
)
ERROR_STATE_KEYWORDS = (
    "something went wrong",
    "try again",
    "couldn't generate",
    "unable to generate",
    "not enough credits",
    "out of credits",
)
SETTINGS_MENU_SELECTORS = [
    "button[aria-haspopup='menu']",
    "button[id*='radix']",
]
MODEL_CONTROL_SELECTORS = [
    "button[role='combobox']:has-text('Veo')",
    "select:has-text('Veo')",
]
MODEL_OPTION_SELECTORS = [
    "[role='option']:has-text('Veo 3.1 - Fast')",
    "option:has-text('Veo 3.1 - Fast')",
]


def _safe_slug(value: str) -> str:
    return "".join(c for c in value if c.isalnum() or c in (" ", "-", "_")).strip()[:120] or "untitled_project"


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _word_count(text: str) -> int:
    return len(_tokenize_words(text))


def _sentence_word_counts(text: str) -> List[int]:
    sentence_parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [_word_count(part) for part in sentence_parts]


def _important_tokens(text: str, min_len: int = 4) -> List[str]:
    tokens = _tokenize_words(text)
    return [t for t in tokens if len(t) >= min_len and t not in STOPWORDS]


def _normalize_string_list(value) -> List[str]:
    if isinstance(value, list):
        return [_normalize_space(str(item)) for item in value if _normalize_space(str(item))]
    if isinstance(value, str):
        split_values = re.split(r"[,\n]", value)
        return [_normalize_space(item) for item in split_values if _normalize_space(item)]
    return []


def _check_visual_overlap(voice_text: str, visual: str, visual_goal: str = "") -> tuple[bool, str, int]:
    voice_tokens = set(_important_tokens(voice_text, min_len=3))
    visual_tokens = set(_important_tokens(f"{visual_goal} {visual}", min_len=3))
    overlap = voice_tokens.intersection(visual_tokens)
    required_overlap = 1 if len(voice_tokens) <= 4 else 2
    if len(overlap) < required_overlap:
        return False, (
            f"visual/script overlap too low (found {len(overlap)}, required {required_overlap})"
        ), len(overlap)
    return True, "ok", len(overlap)


def _check_voice_density(voice_text: str, clip_number: int) -> List[str]:
    errors = []
    words = _word_count(voice_text)
    if words < MIN_WORD_COUNT or words > MAX_WORD_COUNT:
        errors.append(
            f"Clip {clip_number}: word count {words} is outside {MIN_WORD_COUNT}-{MAX_WORD_COUNT}."
        )

    sentence_counts = _sentence_word_counts(voice_text)
    if not sentence_counts:
        errors.append(f"Clip {clip_number}: must contain at least one sentence.")
    elif len(sentence_counts) > MAX_SENTENCE_COUNT:
        errors.append(
            f"Clip {clip_number}: has {len(sentence_counts)} sentences (max {MAX_SENTENCE_COUNT})."
        )

    return errors


def _check_sync_terms(voice_text: str, visual: str, visual_goal: str, sync_terms: List[str], clip_number: int) -> List[str]:
    errors = []
    if len(sync_terms) < 2:
        errors.append(f"Clip {clip_number}: sync_terms must include at least 2 items.")
        return errors

    combined_visual = f"{visual_goal} {visual}"
    voice_tokens = set(_important_tokens(voice_text, min_len=3))
    visual_tokens = set(_important_tokens(combined_visual, min_len=3))

    matched_visual_terms = 0
    matched_voice_terms = 0
    for term in sync_terms:
        term_tokens = set(_important_tokens(term, min_len=3))
        if term_tokens.intersection(visual_tokens):
            matched_visual_terms += 1
        if term_tokens.intersection(voice_tokens):
            matched_voice_terms += 1

    required_visual_matches = max(1, min(2, len(sync_terms)))
    if matched_visual_terms < required_visual_matches:
        errors.append(
            f"Clip {clip_number}: sync_terms are not grounded strongly enough in the visual plan."
        )
    if matched_voice_terms == 0:
        errors.append(
            f"Clip {clip_number}: sync_terms do not connect back to the narration."
        )
    return errors


def _check_clip_role_sequence(items: List[dict]) -> List[str]:
    errors = []
    expected_roles = EXPECTED_CLIP_ROLE_SEQUENCES.get(len(items))
    if not expected_roles:
        return [f"Expected {MIN_CLIP_COUNT}-{MAX_CLIP_COUNT} clips, received {len(items)}."]

    for idx, expected_role in enumerate(expected_roles, start=1):
        actual_role = _normalize_space(str(items[idx - 1].get("clip_role", "")))
        if actual_role != expected_role:
            errors.append(
                f"Clip {idx}: clip_role must be '{expected_role}', found '{actual_role}'."
            )
    return errors


def _adjacent_continuity_overlap(previous_text: str, current_text: str) -> int:
    previous_tokens = set(_important_tokens(previous_text, min_len=3))
    current_tokens = set(_important_tokens(current_text, min_len=3))
    return len(previous_tokens.intersection(current_tokens))


def _check_consistency(items: List[dict]) -> List[str]:
    errors = []
    if not items:
        return errors

    base_voice = json.dumps(items[0].get("voice", {}), sort_keys=True)
    base_audio = json.dumps(items[0].get("background_audio", {}), sort_keys=True)
    for idx, item in enumerate(items[1:], start=2):
        current_voice = json.dumps(item.get("voice", {}), sort_keys=True)
        current_audio = json.dumps(item.get("background_audio", {}), sort_keys=True)
        if current_voice != base_voice:
            errors.append(f"Clip {idx}: voice object differs from clip 1.")
        if current_audio != base_audio:
            errors.append(f"Clip {idx}: background_audio object differs from clip 1.")
    return errors


def _write_quality_report(project_name: str, report: dict) -> None:
    safe_project = _safe_slug(project_name or "untitled_project")
    output_dir = os.path.join("outputs", safe_project)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _project_output_dir(project_name: str) -> str:
    safe_project = _safe_slug(project_name or "untitled_project")
    output_dir = os.path.join("outputs", safe_project)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _read_locator_text(locator) -> str:
    try:
        value = locator.evaluate(
            """(el) => {
                if ('value' in el && typeof el.value === 'string') {
                    return el.value;
                }
                return (el.innerText || el.textContent || '').trim();
            }"""
        )
        return _normalize_space(value)
    except Exception:
        return ""


def _locator_is_enabled(locator) -> bool:
    try:
        return locator.is_enabled()
    except Exception:
        try:
            return (
                locator.get_attribute("disabled") is None
                and locator.get_attribute("aria-disabled") != "true"
            )
        except Exception:
            return False


def _find_first_visible(page, selectors: List[str], timeout_ms: int, label: str):
    deadline = time.time() + (timeout_ms / 1000)
    last_error = None
    while time.time() < deadline:
        for selector in selectors:
            locator = page.locator(selector).first
            try:
                if locator.count() > 0 and locator.is_visible():
                    return locator, selector
            except Exception as exc:
                last_error = exc
        page.wait_for_timeout(BROWSER_POLL_MS)
    detail = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(f"Could not find visible {label}.{detail}")


def _find_first_enabled(page, selectors: List[str], timeout_ms: int, label: str):
    deadline = time.time() + (timeout_ms / 1000)
    last_error = None
    while time.time() < deadline:
        for selector in selectors:
            locator = page.locator(selector).first
            try:
                if locator.count() > 0 and locator.is_visible() and _locator_is_enabled(locator):
                    return locator, selector
            except Exception as exc:
                last_error = exc
        page.wait_for_timeout(BROWSER_POLL_MS)
    detail = f" Last error: {last_error}" if last_error else ""
    raise RuntimeError(f"Could not find enabled {label}.{detail}")


def _count_matches(page, selectors: List[str]) -> int:
    total = 0
    for selector in selectors:
        try:
            total += page.locator(selector).count()
        except Exception:
            continue
    return total


def _has_visible_match(page, selectors: List[str]) -> bool:
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = min(locator.count(), 5)
            for idx in range(count):
                if locator.nth(idx).is_visible():
                    return True
        except Exception:
            continue
    return False


def _body_text(page) -> str:
    try:
        return _normalize_space(page.locator("body").inner_text(timeout=3000))
    except Exception:
        return ""


def _find_error_keyword(text: str) -> str:
    lowered = (text or "").lower()
    return next((keyword for keyword in ERROR_STATE_KEYWORDS if keyword in lowered), "")


def _snapshot_generation_state(page, prompt_box, generate_btn) -> dict:
    body_text = _body_text(page)
    return {
        "prompt_text": _read_locator_text(prompt_box),
        "button_enabled": _locator_is_enabled(generate_btn),
        "activity_visible": _has_visible_match(page, ACTIVE_GENERATION_SELECTORS),
        "activity_keywords": any(
            keyword in body_text.lower() for keyword in GENERATION_STATE_KEYWORDS
        ),
        "result_count": _count_matches(page, RESULT_TILE_SELECTORS),
        "error_keyword": _find_error_keyword(body_text),
    }


def _submission_signals(before_state: dict, after_state: dict, expected_prompt: str) -> List[str]:
    signals = []
    before_prompt = _normalize_space(before_state.get("prompt_text", ""))
    after_prompt = _normalize_space(after_state.get("prompt_text", ""))
    expected_prompt = _normalize_space(expected_prompt)

    if after_prompt == "":
        signals.append("prompt cleared")
    elif after_prompt != expected_prompt and after_prompt != before_prompt:
        signals.append("prompt changed")

    if before_state.get("button_enabled", True) and not after_state.get("button_enabled", True):
        signals.append("generate button disabled")

    if after_state.get("activity_visible") or after_state.get("activity_keywords"):
        signals.append("generation activity detected")

    if after_state.get("result_count", 0) > before_state.get("result_count", 0):
        signals.append("new result tile detected")

    return signals


def _wait_for_submission_signal(page, prompt_box, generate_btn, expected_prompt: str, before_state: dict, timeout_ms: int = 15000):
    deadline = time.time() + (timeout_ms / 1000)
    last_state = before_state
    while time.time() < deadline:
        current_state = _snapshot_generation_state(page, prompt_box, generate_btn)
        last_state = current_state
        if current_state["error_keyword"]:
            raise RuntimeError(f"Flow reported an error: '{current_state['error_keyword']}'.")
        signals = _submission_signals(before_state, current_state, expected_prompt)
        if signals:
            return current_state, signals
        page.wait_for_timeout(BROWSER_POLL_MS)
    raise RuntimeError(
        "Generate action did not change the Flow UI state. Prompt was never confirmed as submitted."
    )


def _enter_prompt(prompt_box, prompt: str) -> None:
    prompt_box.click(timeout=5000)
    prompt_box.fill("")
    prompt_box.fill(prompt)

    if _read_locator_text(prompt_box) == _normalize_space(prompt):
        return

    prompt_box.click()
    prompt_box.press("Control+A")
    prompt_box.press("Backspace")
    prompt_box.type(prompt, delay=12)

    if _read_locator_text(prompt_box) != _normalize_space(prompt):
        raise RuntimeError("Prompt box did not retain the prompt text after entry.")


def _wait_for_generation_completion(page, baseline_result_count: int, timeout_ms: int = 180000):
    deadline = time.time() + (timeout_ms / 1000)
    saw_activity = False
    stable_result_cycles = 0

    while time.time() < deadline:
        body_text = _body_text(page)
        error_keyword = _find_error_keyword(body_text)
        if error_keyword:
            raise RuntimeError(f"Flow reported an error: '{error_keyword}'.")

        activity_visible = _has_visible_match(page, ACTIVE_GENERATION_SELECTORS)
        activity_keywords = any(
            keyword in body_text.lower() for keyword in GENERATION_STATE_KEYWORDS
        )
        result_count = _count_matches(page, RESULT_TILE_SELECTORS)

        if activity_visible or activity_keywords:
            saw_activity = True

        if result_count > baseline_result_count and not activity_visible and not activity_keywords:
            stable_result_cycles += 1
        else:
            stable_result_cycles = 0

        if saw_activity and not activity_visible and not activity_keywords:
            return "generation activity completed"

        if stable_result_cycles >= 3:
            return "new result tile appeared"

        page.wait_for_timeout(1000)

    raise RuntimeError("Timed out waiting for Flow to finish generation for this clip.")


def _capture_browser_artifacts(page, project_name: str, artifact_name: str) -> str:
    output_dir = _project_output_dir(project_name or "untitled_project")
    screenshot_path = os.path.join(output_dir, f"{artifact_name}.png")
    html_path = os.path.join(output_dir, f"{artifact_name}.html")

    try:
        page.screenshot(path=screenshot_path, full_page=True)
    except Exception:
        screenshot_path = ""

    try:
        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(page.content())
    except Exception:
        html_path = ""

    saved_paths = [path for path in (screenshot_path, html_path) if path]
    if not saved_paths:
        return "no debug artifacts captured"
    return ", ".join(saved_paths)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_flow_editor_ready(page):
    try:
        prompt_box, prompt_selector = _find_first_visible(
            page,
            PROMPT_BOX_SELECTORS,
            timeout_ms=8000,
            label="prompt box",
        )
        return prompt_box, prompt_selector, "existing_editor"
    except Exception:
        new_project_btn, selector_used = _find_first_enabled(
            page,
            NEW_PROJECT_SELECTORS,
            timeout_ms=20000,
            label="New project button",
        )
        new_project_btn.click()
        page.wait_for_timeout(2500)
        prompt_box, prompt_selector = _find_first_visible(
            page,
            PROMPT_BOX_SELECTORS,
            timeout_ms=20000,
            label="prompt box",
        )
        return prompt_box, prompt_selector, f"new_project:{selector_used}"


def _locator_is_selected(locator) -> bool:
    try:
        return (
            locator.get_attribute("aria-selected") == "true"
            or locator.get_attribute("data-state") == "active"
            or locator.get_attribute("aria-pressed") == "true"
        )
    except Exception:
        return False


def _ensure_selected(locator, label: str, page=None, timeout_ms: int = 1000) -> None:
    if not locator.is_visible():
        raise RuntimeError(f"Could not verify {label}: control is not visible.")
    if _locator_is_selected(locator):
        return
    locator.click()
    if page is not None:
        page.wait_for_timeout(timeout_ms)
    if not _locator_is_selected(locator):
        raise RuntimeError(f"Could not verify {label}: control did not become active.")


def _verify_flow_settings(page) -> None:
    settings_btn = None
    last_error = None
    for selector in SETTINGS_MENU_SELECTORS:
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

    if settings_btn.get_attribute("data-state") != "open":
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

        _ensure_selected(video_tab, "Video mode", page=page)
        _ensure_selected(portrait_tab, "Portrait mode", page=page)
        _ensure_selected(x2_tab, "x2 duration", page=page)

        model_control = None
        for selector in MODEL_CONTROL_SELECTORS:
            candidate = page.locator(selector).first
            try:
                if candidate.count() > 0 and candidate.is_visible():
                    model_control = candidate
                    break
            except Exception:
                continue
        if model_control is None:
            raise RuntimeError("Could not locate the Flow model selector.")

        model_text = _normalize_space(_read_locator_text(model_control))
        if "Veo 3.1 - Fast" not in model_text:
            model_control.click()
            page.wait_for_timeout(1000)
            selected = False
            for selector in MODEL_OPTION_SELECTORS:
                option = page.locator(selector).first
                try:
                    if option.count() > 0 and option.is_visible():
                        option.click()
                        selected = True
                        break
                except Exception:
                    continue
            if not selected:
                raise RuntimeError("Could not locate the 'Veo 3.1 - Fast' model option.")
            page.wait_for_timeout(1000)
            model_text = _normalize_space(_read_locator_text(model_control))
            if "Veo 3.1 - Fast" not in model_text:
                raise RuntimeError("Model selector did not confirm 'Veo 3.1 - Fast'.")
    finally:
        try:
            page.keyboard.press("Escape")
            page.wait_for_timeout(500)
        except Exception:
            pass


def _build_cinematic_prompt(item: dict, clip_number: int = 1) -> str:
    """Build a compact Flow prompt that keeps narration and visuals tightly synced."""
    video_style = str(item.get("video_style", "cinematic documentary")).strip()
    visual = str(item.get("visual", "")).strip()
    voice_text = str(item.get("voice_text", "")).strip()
    clip_role = str(item.get("clip_role", "Clip")).strip()
    visual_goal = str(item.get("visual_goal", "")).strip()
    sync_terms = _normalize_string_list(item.get("sync_terms", []))
    voice = item.get("voice", {})
    audio = item.get("background_audio", {})

    audio_type = audio.get("type")
    if not audio_type:
        raise ValueError(
            f"CRITICAL: background_audio.type missing from clip {clip_number} JSON. "
            "This field is mandatory for synced audio intent."
        )

    narrator_tone = str(voice.get("tone", "warm, conversational")).strip()
    narrator_gender = str(voice.get("gender", "male")).strip()
    sfx_layers = audio.get("sfx_layers", audio_type)
    sync_terms_text = ", ".join(sync_terms)

    prompt = (
        f"A {narrator_tone.lower()} {narrator_gender.lower()} narrator says: {voice_text}. "
        f"Clip role: {clip_role}. "
        f"Viewer takeaway for this shot: {visual_goal}. "
        f"Must visibly include: {sync_terms_text}. "
        f"Visual requirement: {visual}. "
        f"Style: {video_style}. "
        f"Ambient sound bed: {sfx_layers}. "
        "Portrait 9:16. Keep the visual literal, educational, and continuous with adjacent clips."
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

    def _run(self, url: str, prompts: List[str] = None, json_content: str = None, project_name: str = None, dry_run: bool = False) -> str:
        auth_file = "auth.json"
        
        # --- Logic to extract prompts from JSON if provided ---
        final_prompts = []
        if json_content:
            try:
                required_keys = [
                    "clip_role",
                    "sync_terms",
                    "visual_goal",
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

                def normalize_item(item: dict, clip_number: int = 1) -> dict:
                    missing = [k for k in required_keys if k not in item]
                    if missing:
                        raise ValueError(f"Clip {clip_number}: missing required keys: {', '.join(missing)}")

                    voice_text = str(item["voice_text"]).strip()
                    if not voice_text:
                        raise ValueError(f"Clip {clip_number}: 'voice_text' is empty.")

                    density_errors = _check_voice_density(voice_text, clip_number)
                    if density_errors:
                        raise ValueError(" | ".join(density_errors))

                    background_audio = item.get("background_audio", {})
                    if not isinstance(background_audio, dict):
                        raise ValueError(f"Clip {clip_number}: background_audio must be an object.")
                    if "type" not in background_audio and "audio_type" in background_audio:
                        background_audio["type"] = background_audio["audio_type"]
                    if "volume" not in background_audio and "audio_volume" in background_audio:
                        background_audio["volume"] = background_audio["audio_volume"]
                    if isinstance(background_audio.get("sfx_layers"), list):
                        background_audio["sfx_layers"] = ", ".join(str(x) for x in background_audio["sfx_layers"])
                    missing_audio_keys = [
                        key for key in REQUIRED_BACKGROUND_AUDIO_KEYS if key not in background_audio
                    ]
                    if missing_audio_keys:
                        raise ValueError(
                            f"Clip {clip_number}: background_audio missing keys: {', '.join(missing_audio_keys)}."
                        )
                    if background_audio.get("generate_with_video") is not True:
                        raise ValueError(
                            f"Clip {clip_number}: background_audio.generate_with_video must be true."
                        )

                    if str(item["orientation"]).strip().lower() != "portrait":
                        raise ValueError(f"Clip {clip_number}: orientation must be 'portrait'.")
                    if str(item["aspect_ratio"]).strip() != "9:16":
                        raise ValueError(f"Clip {clip_number}: aspect_ratio must be '9:16'.")

                    clip_role = _normalize_space(str(item["clip_role"]))
                    if not clip_role:
                        raise ValueError(f"Clip {clip_number}: clip_role is empty.")

                    sync_terms = _normalize_string_list(item.get("sync_terms"))
                    visual_goal = _normalize_space(str(item.get("visual_goal", "")))
                    if not visual_goal:
                        raise ValueError(f"Clip {clip_number}: visual_goal is empty.")

                    overlap_ok, overlap_msg, overlap_count = _check_visual_overlap(
                        voice_text,
                        str(item["visual"]),
                        visual_goal,
                    )
                    if not overlap_ok:
                        raise ValueError(f"Clip {clip_number}: {overlap_msg}.")

                    sync_errors = _check_sync_terms(
                        voice_text,
                        str(item["visual"]),
                        visual_goal,
                        sync_terms,
                        clip_number,
                    )
                    if sync_errors:
                        raise ValueError(" | ".join(sync_errors))

                    normalized = {
                        "clip_role": clip_role,
                        "voice_text": voice_text,
                        "sync_terms": sync_terms,
                        "visual_goal": visual_goal,
                        "voice": item["voice"],
                        "background_audio": background_audio,
                        "visual": item["visual"],
                        "orientation": item["orientation"],
                        "aspect_ratio": item["aspect_ratio"],
                        "video_style": item["video_style"],
                        "_overlap_count": overlap_count,
                    }
                    if "clip_label" in item:
                        normalized["clip_label"] = item["clip_label"]
                    else:
                        normalized["clip_label"] = f"CLIP {clip_number}"

                    return normalized

                def normalize_all_items(raw_items: List[dict]) -> List[dict]:
                    if len(raw_items) < MIN_CLIP_COUNT or len(raw_items) > MAX_CLIP_COUNT:
                        raise ValueError(
                            f"Expected {MIN_CLIP_COUNT}-{MAX_CLIP_COUNT} clips, received {len(raw_items)}."
                        )
                    if role_errors := _check_clip_role_sequence(raw_items):
                        raise ValueError(" | ".join(role_errors))

                    normalized_items = []
                    for idx, clip in enumerate(raw_items, start=1):
                        if not isinstance(clip, dict):
                            raise ValueError(f"Clip {idx}: each list item must be an object.")
                        normalized_items.append(normalize_item(clip, idx))
                    if consistency_errors := _check_consistency(normalized_items):
                        raise ValueError(" | ".join(consistency_errors))
                    return normalized_items

                normalized_items = []
                if isinstance(data, list):
                    normalized_items = normalize_all_items(data)
                else:
                    raise ValueError("json_content must decode to a list of 5 or 6 clip objects.")

                quality_report = {
                    "timestamp": _utc_timestamp(),
                    "project_name": project_name or "untitled_project",
                    "status": "passed",
                    "clip_count": len(normalized_items),
                    "checks": {
                        "clip_count_range": f"{MIN_CLIP_COUNT}-{MAX_CLIP_COUNT}",
                        "word_count_range": f"{MIN_WORD_COUNT}-{MAX_WORD_COUNT}",
                        "max_sentences": MAX_SENTENCE_COUNT,
                        "required_fields": list(required_keys),
                        "flow_settings_target": "Video / Portrait / x2 / Veo 3.1 - Fast",
                    },
                    "clip_metrics": [],
                }

                for clip_index, normalized_item in enumerate(normalized_items, start=1):
                    nl_prompt = _build_cinematic_prompt(normalized_item, clip_index)
                    print(f"   Prompt [{clip_index}]: {nl_prompt[:70]}...")
                    final_prompts.append(nl_prompt)

                    words = _word_count(normalized_item["voice_text"])
                    sentence_count = len(_sentence_word_counts(normalized_item["voice_text"]))
                    continuity_overlap = 0
                    if clip_index > 1:
                        continuity_overlap = _adjacent_continuity_overlap(
                            normalized_items[clip_index - 2]["voice_text"],
                            normalized_item["voice_text"],
                        )
                    quality_report["clip_metrics"].append({
                        "clip": clip_index,
                        "clip_role": normalized_item["clip_role"],
                        "word_count": words,
                        "sentence_count": sentence_count,
                        "overlap_count": normalized_item.get("_overlap_count", 0),
                        "sync_term_count": len(normalized_item["sync_terms"]),
                        "continuity_overlap": continuity_overlap,
                    })

                _write_quality_report(project_name or "untitled_project", quality_report)
                print(f"OK Extracted {len(final_prompts)} prompts from JSON content.")
            except Exception as e:
                failure_report = {
                    "timestamp": _utc_timestamp(),
                    "project_name": project_name or "untitled_project",
                    "status": "failed",
                    "error": str(e),
                }
                _write_quality_report(project_name or "untitled_project", failure_report)
                raise RuntimeError(
                    "PRE-FLIGHT VALIDATION FAILED: "
                    f"{e}. Regenerate ONLY the failing clip(s) while preserving passing clips."
                ) from e
        
        # Merge with direct prompts if any (though usually it's one or the other)
        if prompts:
            final_prompts.extend(prompts)
            
        if not final_prompts:
            return "❌ Error: No prompts found. Please provide either 'prompts' list or valid 'json_content' containing 'visual' fields."

        if dry_run:
            return (
                f"DRY RUN PASSED: validated {len(final_prompts)} clips and built "
                f"{len(final_prompts)} Flow prompts. Browser launch skipped, no Flow credits spent."
            )

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
                page.goto(url, timeout=120000, wait_until="domcontentloaded")
                page.wait_for_timeout(5000)
                
                # Check if we are actually logged in.
                body_text = _body_text(page).lower()
                if "sign in" in page.title().lower() or "sign in" in body_text or "login" in body_text:
                     return "❌ Error: It seems you are not logged in. Please run 'python setup_auth.py' again."

                # --- STEP 1: Ensure the editor is ready ---
                print("🔎 Checking whether the Flow editor is already open...")
                prompt_box, prompt_selector, editor_source = _ensure_flow_editor_ready(page)
                if editor_source == "existing_editor":
                    print("✅ Prompt box is already available")
                else:
                    print("🆕 Prompt box not visible yet, opening a new project...")
                    selector_used = editor_source.split("new_project:", 1)[1]
                    print(f"✅ Clicked 'New project' via selector: {selector_used}")
                print(f"✅ Flow editor ready via prompt box selector: {prompt_selector}")

                # --- STEP 1.5 & 1.6: Verify Mode and Settings via Settings Menu ---
                print("\n⚙️ Verifying generation settings...")
                _verify_flow_settings(page)
                print("   ✅ Flow settings verified: Video / Portrait / x2 / Veo 3.1 - Fast")

                # --- STEP 2: Process Prompts ---
                for i, prompt in enumerate(final_prompts):
                    clip_num = i + 1
                    print(f"\n📌 Processing Clip {clip_num}/{len(final_prompts)}: '{prompt[:30]}...'")
                    
                    # Locate Prompt Box
                    print("   🔍 Finding prompt box...")
                    prompt_box, prompt_selector = _find_first_visible(
                        page,
                        PROMPT_BOX_SELECTORS,
                        timeout_ms=15000,
                        label="prompt box"
                    )
                    print(f"   ✅ Using prompt box selector: {prompt_selector}")
                    _enter_prompt(prompt_box, prompt)
                    print("   ✍️ Prompt entered")
                    page.wait_for_timeout(1000)

                    # Locate Generate Button
                    print("   🚀 Clicking Generate...")
                    generate_btn, generate_selector = _find_first_enabled(
                        page,
                        GENERATE_BUTTON_SELECTORS,
                        timeout_ms=15000,
                        label="Generate button"
                    )
                    print(f"   ✅ Using generate selector: {generate_selector}")
                    
                    # --- STEP 3: Generate with Retry Logic ---
                    MAX_RETRIES = 2
                    clip_succeeded = False
                    last_attempt_error = ""
                    
                    for attempt in range(MAX_RETRIES):
                        attempt_label = f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        try:
                            if attempt > 0:
                                print(f"   🔄 Retrying clip {clip_num} {attempt_label}...")
                                prompt_box, _ = _find_first_visible(
                                    page,
                                    PROMPT_BOX_SELECTORS,
                                    timeout_ms=15000,
                                    label="prompt box"
                                )
                                _enter_prompt(prompt_box, prompt)
                                generate_btn, generate_selector = _find_first_enabled(
                                    page,
                                    GENERATE_BUTTON_SELECTORS,
                                    timeout_ms=15000,
                                    label="Generate button"
                                )
                                print(f"   ✅ Retrying with generate selector: {generate_selector}")

                            before_state = _snapshot_generation_state(page, prompt_box, generate_btn)
                            generate_btn.click()
                            print(f"   ✅ Generate clicked! {attempt_label}")
                            
                            try:
                                _, submission_signals = _wait_for_submission_signal(
                                    page,
                                    prompt_box,
                                    generate_btn,
                                    prompt,
                                    before_state,
                                    timeout_ms=15000
                                )
                            except RuntimeError as submit_error:
                                if "never confirmed as submitted" not in str(submit_error):
                                    raise
                                print("   ⌨️ Button click was ignored, trying Control+Enter...")
                                prompt_box.click()
                                prompt_box.press("Control+Enter")
                                _, submission_signals = _wait_for_submission_signal(
                                    page,
                                    prompt_box,
                                    generate_btn,
                                    prompt,
                                    before_state,
                                    timeout_ms=15000
                                )
                            print(
                                f"   ✅ Submission confirmed via: {', '.join(submission_signals)}"
                            )

                            print(f"   ⏳ Waiting for generation to complete (timeout 3 mins) {attempt_label}...")
                            completion_reason = _wait_for_generation_completion(
                                page,
                                baseline_result_count=before_state["result_count"],
                                timeout_ms=180000
                            )
                            
                            results.append(
                                f"Clip {clip_num}: ✅ Generated ({completion_reason}; submission confirmed)"
                            )
                            clip_succeeded = True
                            break
                            
                        except Exception as e:
                            last_attempt_error = str(e)
                            print(f"   ⚠️ {attempt_label} failed: {last_attempt_error}")
                            if attempt < MAX_RETRIES - 1:
                                print(f"   ⏳ Waiting 5 seconds before retry...")
                                time.sleep(5)
                    
                    if not clip_succeeded:
                        artifact_paths = _capture_browser_artifacts(
                            page,
                            project_name or "untitled_project",
                            f"clip_{clip_num:02d}_failed"
                        )
                        results.append(
                            f"Clip {clip_num}: ❌ FAILED after {MAX_RETRIES} attempts - "
                            f"{last_attempt_error}. Debug artifacts: {artifact_paths}"
                        )
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
                artifact_paths = _capture_browser_artifacts(
                    page,
                    project_name or "untitled_project",
                    "flow_setup_failed"
                )
                msg = f"❌ Error during automation: {e}. Debug artifacts: {artifact_paths}"
                print(msg)

                results.append(msg)
            
            print(f"\n{'='*60}")
            print("🏁 FINISHED. Browser will remain open for 30 seconds for you to review/download.")
            time.sleep(30)
            
            return "\n".join(results)
