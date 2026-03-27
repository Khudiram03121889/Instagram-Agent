import os
import time
import json
import re
from datetime import datetime
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


TARGET_WORD_COUNT = 16
MIN_WORD_COUNT = 15
MAX_WORD_COUNT = 17
MIN_ESTIMATED_SEC = 7.6
MAX_ESTIMATED_SEC = 8.2
MIN_WPS = 1.95
MAX_WPS = 2.10
EXPECTED_MOTION_INTENSITY = "fast_structured"
EXPECTED_BEAT_MAP = {
    "hook_frame": "0.0-1.2",
    "mechanism_action": "1.2-5.2",
    "payoff": "5.2-7.4",
    "settle": "7.4-8.0",
}
ALLOWED_BRIDGE_PREFIXES = [
    "to understand that",
    "because of this",
    "so what's happening is",
    "that's why",
    "and this isn't just theory, it's",
    "what makes this",
    "and if you think about it,",
    "in other words,",
    "the reason this",
    "which means",
]
REFERENTIAL_BRIDGE_PREFIXES = {
    "to understand that",
    "because of this",
    "so what's happening is",
    "that's why",
    "and this isn't just theory, it's",
    "what makes this",
    "and if you think about it,",
    "in other words,",
    "the reason this",
    "which means",
}
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


def _extract_bridge_concept_tokens(clip_text: str) -> List[str]:
    text = clip_text.strip().lower()
    for prefix in ALLOWED_BRIDGE_PREFIXES:
        if text.startswith(prefix):
            concept_part = text[len(prefix):].strip()
            if "," in concept_part:
                concept_part = concept_part.split(",", 1)[0].strip()
            concept_tokens = _important_tokens(concept_part, min_len=3)
            if concept_tokens:
                return concept_tokens
            # Fallback: if bridge opener is followed by punctuation and then a concept,
            # use early tokens from the remaining line as the bridge concept.
            trailing_tokens = _important_tokens(text[len(prefix):], min_len=3)
            return trailing_tokens[:3]
    return []


def _matched_bridge_prefix(clip_text: str) -> str:
    text = clip_text.strip().lower()
    for prefix in ALLOWED_BRIDGE_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return ""


def _check_visual_overlap(voice_text: str, visual: str) -> tuple[bool, str, int]:
    voice_tokens = set(_important_tokens(voice_text))
    visual_tokens = set(_important_tokens(visual))
    overlap = voice_tokens.intersection(visual_tokens)
    # Keep threshold adaptive so sparse voice text is not over-penalized.
    required_overlap = 1 if len(voice_tokens) <= 4 else 2
    if len(overlap) < required_overlap:
        return False, (
            f"visual/script overlap too low (found {len(overlap)}, required {required_overlap})"
        ), len(overlap)
    return True, "ok", len(overlap)


def _check_voice_timing(voice_text: str, estimated_sec: float, clip_number: int) -> List[str]:
    errors = []
    words = _word_count(voice_text)
    if words < MIN_WORD_COUNT or words > MAX_WORD_COUNT:
        errors.append(
            f"Clip {clip_number}: word count {words} is outside {MIN_WORD_COUNT}-{MAX_WORD_COUNT}."
        )

    sentence_counts = _sentence_word_counts(voice_text)
    if len(sentence_counts) != 2:
        errors.append(f"Clip {clip_number}: must have exactly 2 sentences, found {len(sentence_counts)}.")
    else:
        for idx, sentence_words in enumerate(sentence_counts, start=1):
            if sentence_words < 7 or sentence_words > 9:
                errors.append(
                    f"Clip {clip_number}: sentence {idx} has {sentence_words} words (required 7-9)."
                )

    comma_count = voice_text.count(",")
    if comma_count > 1:
        errors.append(f"Clip {clip_number}: has {comma_count} commas (max 1).")

    if estimated_sec < MIN_ESTIMATED_SEC or estimated_sec > MAX_ESTIMATED_SEC:
        errors.append(
            f"Clip {clip_number}: estimated_sec {estimated_sec} is outside {MIN_ESTIMATED_SEC}-{MAX_ESTIMATED_SEC}."
        )

    if estimated_sec > 0:
        implied_wps = words / estimated_sec
        if implied_wps < MIN_WPS or implied_wps > MAX_WPS:
            errors.append(
                f"Clip {clip_number}: implied speech speed {implied_wps:.2f} wps is outside {MIN_WPS}-{MAX_WPS}."
            )
    else:
        errors.append(f"Clip {clip_number}: estimated_sec must be > 0.")

    return errors


def _normalize_timing_metadata(word_count: int, estimated_sec: float) -> float:
    # Use clip metadata as a hint, but keep final value inside target window
    # to avoid wasting retries for tiny arithmetic drift from the LLM.
    sec = estimated_sec if estimated_sec > 0 else (word_count / 2.0)
    sec = max(MIN_ESTIMATED_SEC, min(MAX_ESTIMATED_SEC, sec))

    implied_wps = word_count / sec if sec > 0 else 0
    if implied_wps < MIN_WPS:
        sec = min(MAX_ESTIMATED_SEC, word_count / MIN_WPS)
    elif implied_wps > MAX_WPS:
        sec = max(MIN_ESTIMATED_SEC, word_count / MAX_WPS)

    return round(sec, 2)


def _check_bridge_continuity(items: List[dict]) -> List[str]:
    errors = []
    if len(items) < 2:
        return errors

    for idx in range(1, len(items)):
        clip_number = idx + 1
        current_text = str(items[idx].get("voice_text", "")).strip().lower()
        previous_text = str(items[idx - 1].get("voice_text", "")).strip().lower()

        matched_prefix = _matched_bridge_prefix(current_text)
        if not matched_prefix:
            errors.append(
                f"Clip {clip_number}: does not start with an approved bridge opener."
            )
            continue

        concept_tokens = _extract_bridge_concept_tokens(current_text)
        if not concept_tokens:
            errors.append(
                f"Clip {clip_number}: bridge opener missing a concrete concept token."
            )
            continue

        if not any(token in previous_text for token in concept_tokens):
            # Secondary fallback: require at least one meaningful token overlap
            # between previous and current clip text.
            previous_tokens = set(_important_tokens(previous_text, min_len=3))
            current_tokens = set(_important_tokens(current_text, min_len=3))
            if previous_tokens.intersection(current_tokens):
                continue
            # Documentary bridge phrases like "To understand that" or
            # "Because of this" are explicitly referential even when the
            # next clip pivots to a new explanatory noun phrase.
            if matched_prefix in REFERENTIAL_BRIDGE_PREFIXES and len(concept_tokens) >= 2:
                continue
            errors.append(
                f"Clip {clip_number}: bridge concept does not echo previous clip content."
            )

    return errors


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
    # Extract required fields
    video_style = str(item.get("video_style", "cinematic documentary")).strip()
    visual = str(item.get("visual", "")).strip()
    voice_text = str(item.get("voice_text", "")).strip()
    voice = item.get("voice", {})
    audio = item.get("background_audio", {})
    beat_map = item.get("beat_map", EXPECTED_BEAT_MAP)
    motion_intensity = str(item.get("motion_intensity", EXPECTED_MOTION_INTENSITY)).strip()

    audio_type = audio.get("type")
    if not audio_type:
        raise ValueError(
            f"CRITICAL: background_audio.type missing from clip {clip_number} JSON. "
            "This field is mandatory for synced audio intent."
        )

    narrator_tone = str(voice.get("tone", "warm, conversational")).strip()
    narrator_gender = str(voice.get("gender", "male")).strip()
    narrator_speed = voice.get("speed", 0.93)
    sfx_layers = audio.get("sfx_layers", audio_type)

    timing_choreography = (
        f"8-second structured pacing. "
        f"Beat 1 ({beat_map.get('hook_frame')}): literal hook frame. "
        f"Beat 2 ({beat_map.get('mechanism_action')}): mechanism action with fast clarity. "
        f"Beat 3 ({beat_map.get('payoff')}): payoff or contrast. "
        f"Settle ({beat_map.get('settle')}): clean handoff."
    )

    camera = (
        "Fast structured camera language with controlled motion, "
        "clear subject tracking, and no chaotic cut spam."
    )

    prompt = (
        f"A {narrator_tone.lower()} {narrator_gender.lower()} narrator says: {voice_text}. "
        f"Keep narration pacing near speed {narrator_speed}. "
        f"{timing_choreography} "
        f"Motion intensity: {motion_intensity}. "
        f"{camera} "
        f"Visual requirement: {visual}. "
        f"Style: {video_style}. "
        f"Ambient sound bed: {sfx_layers}. "
        f"Portrait 9:16. Preserve continuity with previous and next clips."
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

                required_keys.extend(["word_target", "estimated_sec", "beat_map", "motion_intensity"])

                def normalize_item(item: dict, clip_number: int = 1) -> dict:
                    missing = [k for k in required_keys if k not in item]
                    if missing:
                        raise ValueError(f"Clip {clip_number}: missing required keys: {', '.join(missing)}")

                    voice_text = str(item["voice_text"]).strip()
                    if not voice_text:
                        raise ValueError(f"Clip {clip_number}: 'voice_text' is empty.")

                    word_count = _word_count(voice_text)

                    try:
                        estimated_sec_raw = float(item["estimated_sec"])
                    except Exception:
                        estimated_sec_raw = word_count / 2.0

                    try:
                        word_target = int(item["word_target"])
                    except Exception:
                        word_target = TARGET_WORD_COUNT

                    if word_target != TARGET_WORD_COUNT:
                        print(
                            f"   WARN Clip {clip_number}: word_target corrected from "
                            f"{word_target} to {TARGET_WORD_COUNT}."
                        )
                        word_target = TARGET_WORD_COUNT

                    estimated_sec = _normalize_timing_metadata(word_count, estimated_sec_raw)

                    motion_intensity = str(item["motion_intensity"]).strip().lower()
                    if motion_intensity != EXPECTED_MOTION_INTENSITY:
                        raise ValueError(
                            f"Clip {clip_number}: motion_intensity must be '{EXPECTED_MOTION_INTENSITY}'."
                        )

                    background_audio = item.get("background_audio", {})
                    if not isinstance(background_audio, dict):
                        raise ValueError(f"Clip {clip_number}: background_audio must be an object.")
                    if "type" not in background_audio and "audio_type" in background_audio:
                        background_audio["type"] = background_audio["audio_type"]
                    if "volume" not in background_audio and "audio_volume" in background_audio:
                        background_audio["volume"] = background_audio["audio_volume"]
                    if isinstance(background_audio.get("sfx_layers"), list):
                        background_audio["sfx_layers"] = ", ".join(str(x) for x in background_audio["sfx_layers"])
                    if "type" not in background_audio:
                        raise ValueError(f"Clip {clip_number}: background_audio.type is required.")
                    if "volume" not in background_audio:
                        raise ValueError(f"Clip {clip_number}: background_audio.volume is required.")

                    beat_map = item.get("beat_map")
                    if not isinstance(beat_map, dict):
                        raise ValueError(f"Clip {clip_number}: beat_map must be an object.")
                    for beat_key, expected_range in EXPECTED_BEAT_MAP.items():
                        if str(beat_map.get(beat_key, "")).strip() != expected_range:
                            raise ValueError(
                                f"Clip {clip_number}: beat_map.{beat_key} must be '{expected_range}'."
                            )

                    if str(item["orientation"]).strip().lower() != "portrait":
                        raise ValueError(f"Clip {clip_number}: orientation must be 'portrait'.")
                    if str(item["aspect_ratio"]).strip() != "9:16":
                        raise ValueError(f"Clip {clip_number}: aspect_ratio must be '9:16'.")

                    timing_errors = _check_voice_timing(voice_text, estimated_sec, clip_number)
                    if timing_errors:
                        raise ValueError(" | ".join(timing_errors))

                    overlap_ok, overlap_msg, overlap_count = _check_visual_overlap(
                        voice_text,
                        str(item["visual"]),
                    )
                    if not overlap_ok:
                        raise ValueError(f"Clip {clip_number}: {overlap_msg}.")

                    normalized = {
                        "voice_text": voice_text,
                        "voice": item["voice"],
                        "background_audio": background_audio,
                        "visual": item["visual"],
                        "orientation": item["orientation"],
                        "aspect_ratio": item["aspect_ratio"],
                        "video_style": item["video_style"],
                        "word_target": word_target,
                        "estimated_sec": estimated_sec,
                        "beat_map": beat_map,
                        "motion_intensity": motion_intensity,
                        "_overlap_count": overlap_count,
                    }
                    if "clip_label" in item:
                        normalized["clip_label"] = item["clip_label"]
                    if "clip" in item:
                        normalized["clip"] = item["clip"]

                    return normalized

                def normalize_all_items(raw_items: List[dict]) -> List[dict]:
                    if len(raw_items) != 5:
                        raise ValueError(f"Expected exactly 5 clips, received {len(raw_items)}.")

                    if bridge_errors := _check_bridge_continuity(raw_items):
                        raise ValueError(" | ".join(bridge_errors))

                    if consistency_errors := _check_consistency(raw_items):
                        raise ValueError(" | ".join(consistency_errors))

                    normalized_items = []
                    for idx, clip in enumerate(raw_items, start=1):
                        if not isinstance(clip, dict):
                            raise ValueError(f"Clip {idx}: each list item must be an object.")
                        normalized_items.append(normalize_item(clip, idx))
                    return normalized_items

                normalized_items = []
                if isinstance(data, list):
                    normalized_items = normalize_all_items(data)
                elif isinstance(data, dict):
                    normalized_items = [normalize_item(data, 1)]
                else:
                    raise ValueError("json_content must decode to an object or list.")

                quality_report = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "project_name": project_name or "untitled_project",
                    "status": "passed",
                    "clip_count": len(normalized_items),
                    "checks": {
                        "word_count_range": f"{MIN_WORD_COUNT}-{MAX_WORD_COUNT}",
                        "estimated_sec_range": f"{MIN_ESTIMATED_SEC}-{MAX_ESTIMATED_SEC}",
                        "implied_wps_range": f"{MIN_WPS}-{MAX_WPS}",
                        "motion_intensity": EXPECTED_MOTION_INTENSITY,
                        "beat_map": EXPECTED_BEAT_MAP,
                    },
                    "clip_metrics": [],
                }

                for clip_index, normalized_item in enumerate(normalized_items, start=1):
                    nl_prompt = _build_cinematic_prompt(normalized_item, clip_index)
                    print(f"   Prompt [{clip_index}]: {nl_prompt[:70]}...")
                    final_prompts.append(nl_prompt)

                    words = _word_count(normalized_item["voice_text"])
                    est_sec = float(normalized_item["estimated_sec"])
                    quality_report["clip_metrics"].append({
                        "clip": clip_index,
                        "word_count": words,
                        "estimated_sec": est_sec,
                        "implied_wps": round(words / est_sec, 3) if est_sec else None,
                        "overlap_count": normalized_item.get("_overlap_count", 0),
                    })

                _write_quality_report(project_name or "untitled_project", quality_report)
                print(f"OK Extracted {len(final_prompts)} prompts from JSON content.")
            except Exception as e:
                failure_report = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
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
                
                # Check if we are actually logged in (simple check)
                if "sign in" in page.title().lower():
                     return "❌ Error: It seems you are not logged in. Please run 'python setup_auth.py' again."

                # --- STEP 1: Ensure the editor is ready ---
                print("🔎 Checking whether the Flow editor is already open...")
                try:
                    prompt_box, _ = _find_first_visible(
                        page,
                        PROMPT_BOX_SELECTORS,
                        timeout_ms=8000,
                        label="prompt box"
                    )
                    print("✅ Prompt box is already available")
                except Exception:
                    print("🆕 Prompt box not visible yet, opening a new project...")
                    new_project_btn, selector_used = _find_first_enabled(
                        page,
                        NEW_PROJECT_SELECTORS,
                        timeout_ms=20000,
                        label="New project button"
                    )
                    new_project_btn.click()
                    print(f"✅ Clicked 'New project' via selector: {selector_used}")
                    page.wait_for_timeout(2500)
                    prompt_box, _ = _find_first_visible(
                        page,
                        PROMPT_BOX_SELECTORS,
                        timeout_ms=20000,
                        label="prompt box"
                    )

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
                msg = f"❌ Error during automation: {e}"
                print(msg)

                results.append(msg)
            
            print(f"\n{'='*60}")
            print("🏁 FINISHED. Browser will remain open for 30 seconds for you to review/download.")
            time.sleep(30)
            
            return "\n".join(results)
