import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

# Force UTF-8 encoding for stdout and stderr to handle emoji on Windows.
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import yaml
from dotenv import load_dotenv
from crewai import Agent, Crew, LLM, Process, Task
from playwright.sync_api import sync_playwright

from tools.browser_tool import VideoGenerationTool
from tools.topic_classifier import classify_topic, format_profile_for_agent


load_dotenv()


BLUEPRINT_ROLE_SEQUENCES = {
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

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}

HOOK_HINTS = {
    "you",
    "your",
    "ever",
    "today",
    "noticed",
    "feel",
    "felt",
    "after",
    "when",
    "while",
    "watch",
    "see",
    "hear",
    "touch",
    "rub",
    "wake",
    "woke",
}

CONTRAST_HINTS = {
    "but",
    "instead",
    "without",
    "actually",
    "rather",
    "looks",
    "seems",
    "yet",
    "even though",
    "not",
}

TAKEAWAY_HINTS = {
    "you",
    "your",
    "means",
    "so",
    "that is why",
    "which means",
    "this is why",
}

TEASER_PHRASES = (
    "more to come",
    "deeper stories",
    "hidden stories",
    "still hide",
    "waiting to be discovered",
    "more secrets",
    "next video",
    "part 2",
)

BANNED_SCRIPT_WORDS = {
    "adrift",
    "ethereal",
    "visceral",
    "cerebral",
    "imperceptible",
    "immense",
    "resonance",
}

BANNED_SCRIPT_PHRASES = (
    "cosmic whisper",
    "change everything",
    "works like a map",
    "hidden clues",
    "mysteries may be hiding",
    "ancient living history",
    "quiet cosmic drift",
    "silent cosmic drift",
)

BORING_FILLER_PHRASES = (
    "more often than you think",
    "people do not realize",
    "people don't realize",
    "it may seem simple",
    "it seems simple",
    "it is actually very interesting",
    "this is very interesting",
    "this matters more than people think",
    "you see this all the time",
    "you see this every day",
)


def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config format in {file_path}: expected a YAML mapping."
        )
    return config


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instagram Agent pipeline runner")
    parser.add_argument(
        "--flow-live",
        action="store_true",
        help="Launch Google Flow after local validation. This can spend Flow credits.",
    )
    parser.add_argument(
        "--flow-dry-run",
        action="store_true",
        help="Validate script and prompts locally without launching Google Flow.",
    )
    return parser.parse_args()


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}='{raw}'. Using default {default}.")
        return default


def get_env_reasoning(name: str, default: str) -> str:
    allowed = {"none", "low", "medium", "high"}
    value = (os.getenv(name) or default).strip().lower()
    if value not in allowed:
        print(f"⚠️ Invalid reasoning_effort '{value}' for {name}. Using '{default}'.")
        return default
    return value


def get_env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"Invalid integer for {name}='{raw}'. Using default {default}.")
        return default
    if value < minimum:
        print(f"{name} must be >= {minimum}. Using default {default}.")
        return default
    return value


def get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    print(f"Invalid boolean for {name}='{raw}'. Using default {default}.")
    return default


def get_env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def safe_project_name(value: str) -> str:
    safe = "".join(
        c for c in (value or "untitled_project") if c.isalnum() or c in (" ", "-", "_")
    ).strip()
    return safe[:120] or "untitled_project"


def extract_json_payload(text: str) -> str:
    if not text:
        raise ValueError("No JSON content found.")
    content = text.strip().replace("```json", "").replace("```", "").strip()
    first_object = content.find("{")
    last_object = content.rfind("}")
    if first_object != -1 and last_object != -1 and last_object > first_object:
        return content[first_object : last_object + 1]
    first_list = content.find("[")
    last_list = content.rfind("]")
    if first_list != -1 and last_list != -1 and last_list > first_list:
        return content[first_list : last_list + 1]
    return content


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def word_count(text: str) -> int:
    return len(tokenize_words(text))


def sentence_word_counts(text: str) -> List[int]:
    sentence_parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return [word_count(part) for part in sentence_parts]


def meaningful_tokens(text: str, min_len: int = 3) -> List[str]:
    tokens = tokenize_words(text)
    return [token for token in tokens if len(token) >= min_len and token not in STOPWORDS]


def overlap_count(text: str, reference: str) -> int:
    return len(set(meaningful_tokens(text)).intersection(meaningful_tokens(reference)))


def list_overlap_count(text: str, reference_items: List[str]) -> int:
    reference_tokens: set[str] = set()
    for item in reference_items:
        reference_tokens.update(meaningful_tokens(str(item)))
    return len(set(meaningful_tokens(text)).intersection(reference_tokens))


def has_phrase(text: str, phrases: Tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def extract_failed_clips(text: str) -> List[int]:
    if not text:
        return []
    return sorted({int(match) for match in re.findall(r"Clip\s+(\d+)", text)})


def extract_script_title_and_clips(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    if not text:
        return "", []

    normalized = text.replace("\r\n", "\n")
    title_match = re.search(r"^\s*Title:\s*(.+)$", normalized, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    clip_pattern = re.compile(r"^\s*Clip\s+(\d+)\s*:\s*", flags=re.MULTILINE)
    matches = list(clip_pattern.finditer(normalized))
    clips: List[Dict[str, Any]] = []

    for idx, match in enumerate(matches):
        clip_number = int(match.group(1))
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        clip_text = normalized[match.end() : next_start].strip()
        clip_text = re.sub(r"\s+", " ", clip_text).strip()
        if clip_text:
            clips.append({"clip": clip_number, "voice_text": clip_text})

    return title, clips


def format_canonical_script(title: str, clips: List[Dict[str, Any]]) -> str:
    title_line = f"Title: {title or 'Untitled'}"
    clip_lines = [title_line, ""]
    for clip in clips:
        clip_lines.append(f"Clip {clip['clip']}:")
        clip_lines.append(clip["voice_text"])
        clip_lines.append("")
    return "\n".join(clip_lines).strip()


def parse_story_blueprint(text: str) -> Dict[str, Any]:
    payload = extract_json_payload(text)
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("Story blueprint must be a JSON object.")
    return data


def format_canonical_blueprint(blueprint: Dict[str, Any]) -> str:
    return json.dumps(blueprint, ensure_ascii=False, indent=2)


def validate_story_blueprint(
    blueprint: Dict[str, Any], allow_sixth_clip: bool = True
) -> List[str]:
    errors: List[str] = []

    topic_angle = str(blueprint.get("topic_angle", "")).strip()
    if not topic_angle:
        errors.append("Blueprint: topic_angle is required.")

    try:
        clip_count = int(blueprint.get("clip_count"))
    except Exception:
        clip_count = -1
    allowed_counts = {5, 6} if allow_sixth_clip else {5}
    if clip_count not in allowed_counts:
        errors.append(
            f"Blueprint: clip_count must be one of {sorted(allowed_counts)}, found {blueprint.get('clip_count')}."
        )
        return errors

    clips = blueprint.get("clips")
    if not isinstance(clips, list):
        errors.append("Blueprint: clips must be a list.")
        return errors

    if len(clips) != clip_count:
        errors.append(
            f"Blueprint: clip_count={clip_count} but clips list has {len(clips)} items."
        )
        return errors

    expected_roles = BLUEPRINT_ROLE_SEQUENCES[clip_count]
    required_keys = {
        "clip_number",
        "clip_role",
        "core_idea",
        "bridge_from_previous",
        "next_clip_seed",
        "viewer_takeaway",
        "visual_anchor_terms",
    }

    for idx, clip in enumerate(clips, start=1):
        if not isinstance(clip, dict):
            errors.append(f"Blueprint clip {idx}: must be an object.")
            continue

        missing = sorted(required_keys.difference(clip.keys()))
        if missing:
            errors.append(
                f"Blueprint clip {idx}: missing required keys: {', '.join(missing)}."
            )
            continue

        if clip.get("clip_number") != idx:
            errors.append(
                f"Blueprint clip {idx}: clip_number must be {idx}, found {clip.get('clip_number')}."
            )

        role = str(clip.get("clip_role", "")).strip()
        if role != expected_roles[idx - 1]:
            errors.append(
                f"Blueprint clip {idx}: clip_role must be '{expected_roles[idx - 1]}', found '{role}'."
            )

        for field in ("core_idea", "viewer_takeaway"):
            if not str(clip.get(field, "")).strip():
                errors.append(f"Blueprint clip {idx}: {field} must not be empty.")

        if idx > 1 and not str(clip.get("bridge_from_previous", "")).strip():
            errors.append(
                f"Blueprint clip {idx}: bridge_from_previous must not be empty."
            )

        if idx < clip_count and not str(clip.get("next_clip_seed", "")).strip():
            errors.append(f"Blueprint clip {idx}: next_clip_seed must not be empty.")

        anchors = clip.get("visual_anchor_terms")
        if not isinstance(anchors, list):
            errors.append(f"Blueprint clip {idx}: visual_anchor_terms must be a list.")
            continue
        if len(anchors) < 2 or len(anchors) > 4:
            errors.append(
                f"Blueprint clip {idx}: visual_anchor_terms must contain 2-4 items."
            )
        if any(not str(anchor).strip() for anchor in anchors):
            errors.append(
                f"Blueprint clip {idx}: visual_anchor_terms cannot contain empty values."
            )

    return errors


def validate_script_clips(
    clips: List[Dict[str, Any]], blueprint: Dict[str, Any]
) -> List[str]:
    errors: List[str] = []
    clip_count = int(blueprint.get("clip_count", 0))
    blueprint_clips = blueprint.get("clips", [])

    if len(clips) != clip_count:
        errors.append(f"Expected exactly {clip_count} clips, found {len(clips)}.")
        return errors

    expected_numbers = list(range(1, clip_count + 1))
    actual_numbers = [clip["clip"] for clip in clips]
    if actual_numbers != expected_numbers:
        errors.append(
            f"Expected clip numbering {expected_numbers}, found {actual_numbers}."
        )
        return errors

    for idx, clip in enumerate(clips, start=1):
        voice_text = clip["voice_text"]
        blueprint_clip = blueprint_clips[idx - 1]
        role = str(blueprint_clip.get("clip_role", "")).strip()
        lowered = voice_text.lower()

        if word_count(voice_text) < 8:
            errors.append(f"Clip {idx}: feels too compressed to explain clearly.")
        if word_count(voice_text) > 42:
            errors.append(f"Clip {idx}: feels too dense for a short-form reel.")

        sentence_counts = sentence_word_counts(voice_text)
        if not sentence_counts:
            errors.append(f"Clip {idx}: must contain at least one sentence.")
        elif len(sentence_counts) > 3:
            errors.append(f"Clip {idx}: has {len(sentence_counts)} sentences (max 3).")

        if any(word in tokenize_words(voice_text) for word in BANNED_SCRIPT_WORDS):
            errors.append(f"Clip {idx}: uses overly poetic or vague wording.")
        if has_phrase(voice_text, BANNED_SCRIPT_PHRASES):
            errors.append(f"Clip {idx}: contains a banned vague phrase.")
        if has_phrase(voice_text, BORING_FILLER_PHRASES):
            errors.append(
                f"Clip {idx}: contains generic filler that makes the explanation feel boring."
            )

        if (
            overlap_count(voice_text, str(blueprint_clip.get("core_idea", ""))) == 0
            and list_overlap_count(
                voice_text, list(blueprint_clip.get("visual_anchor_terms", []))
            )
            == 0
        ):
            errors.append(
                f"Clip {idx}: does not clearly express the approved core idea."
            )

        if list_overlap_count(
            voice_text, list(blueprint_clip.get("visual_anchor_terms", []))
        ) == 0:
            errors.append(
                f"Clip {idx}: lacks concrete anchor language that can be visualized clearly."
            )

        if idx == 1:
            if not any(hint in lowered for hint in HOOK_HINTS):
                errors.append(
                    "Clip 1: hook does not feel rooted in a viewer moment or daily experience."
                )
        else:
            previous_seed = str(blueprint_clips[idx - 2].get("next_clip_seed", "")).strip()
            bridge_text = str(blueprint_clip.get("bridge_from_previous", "")).strip()
            if previous_seed and overlap_count(voice_text, previous_seed) == 0:
                errors.append(
                    f"Clip {idx}: does not continue the planted concept from the previous clip."
                )
            elif bridge_text and overlap_count(voice_text, bridge_text) == 0:
                errors.append(
                    f"Clip {idx}: does not echo the planned bridge concept clearly enough."
                )

        if idx < clip_count:
            next_seed = str(blueprint_clip.get("next_clip_seed", "")).strip()
            if next_seed and overlap_count(voice_text, next_seed) == 0:
                errors.append(
                    f"Clip {idx}: does not plant the next clip concept strongly enough."
                )

        if role == "Question" and "?" not in voice_text and not any(
            cue in lowered for cue in ("why", "how", "what makes", "what is")
        ):
            errors.append("Clip 2: question clip must clearly raise the why/how question.")

        if role == "Contrast/Payoff" and not any(
            cue in lowered for cue in CONTRAST_HINTS
        ):
            errors.append(
                f"Clip {idx}: contrast/payoff clip must contain a real reveal or contrast."
            )

        if role == "Personal Takeaway":
            if overlap_count(voice_text, str(blueprint_clip.get("viewer_takeaway", ""))) == 0:
                errors.append(
                    f"Clip {idx}: final takeaway does not match the approved viewer takeaway."
                )
            if not any(cue in lowered for cue in TAKEAWAY_HINTS):
                errors.append(
                    f"Clip {idx}: final clip does not land like a direct takeaway."
                )
            if has_phrase(voice_text, TEASER_PHRASES):
                errors.append(
                    f"Clip {idx}: final clip ends like a teaser instead of a takeaway."
                )

    return errors


def write_pipeline_artifacts(
    project_name: str,
    blueprint_json: str | None = None,
    script_text: str | None = None,
    prompt_json: str | None = None,
) -> None:
    output_dir = os.path.join("outputs", safe_project_name(project_name))
    os.makedirs(output_dir, exist_ok=True)

    if blueprint_json:
        with open(
            os.path.join(output_dir, "story_blueprint.json"),
            "w",
            encoding="utf-8",
        ) as file:
            parsed = json.loads(blueprint_json)
            json.dump(parsed, file, ensure_ascii=False, indent=2)

    if script_text:
        with open(
            os.path.join(output_dir, "validated_script.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write(script_text)

    if prompt_json:
        with open(
            os.path.join(output_dir, "validated_prompts.json"),
            "w",
            encoding="utf-8",
        ) as file:
            parsed = json.loads(prompt_json)
            json.dump(parsed, file, ensure_ascii=False, indent=2)


def run_browser_operator_stage(
    browser_crew: Crew,
    browser_task: Task,
    browser_task_base_description: str,
    video_tool: VideoGenerationTool,
    video_url: str,
    project_name: str,
    prompt_json_output: str,
    flow_dry_run: bool,
) -> Dict[str, Any]:
    browser_task.description = (
        f"{browser_task_base_description}\n\n"
        f"TARGET URL: {video_url}\n"
        f"PROJECT NAME: {project_name}\n"
        f"DRY RUN: {'true' if flow_dry_run else 'false'}\n\n"
        "PROMPT JSON TO USE EXACTLY:\n"
        f"{prompt_json_output}"
    )

    print("⚙️ Running browser video generation task via browser_operator agent...")
    usage_before = getattr(video_tool, "current_usage_count", 0)
    agent_result = None
    agent_output = ""

    try:
        agent_result = browser_crew.kickoff()
        agent_output = (
            browser_task.output.raw
            if getattr(browser_task, "output", None)
            and hasattr(browser_task.output, "raw")
            and browser_task.output.raw
            else str(agent_result)
        )
    except Exception as browser_task_error:
        agent_output = str(browser_task_error)

    usage_after = getattr(video_tool, "current_usage_count", 0)
    if usage_after > usage_before:
        return {
            "agent_result": agent_result,
            "agent_output": agent_output,
            "execution_path": "agent_tool_call",
        }

    print(
        "⚠️ browser_operator did not execute the assigned Flow tool through CrewAI. "
        "Falling back to deterministic tool execution while keeping the same browser stage inputs."
    )
    try:
        deterministic_output = video_tool._run(
            url=video_url,
            json_content=prompt_json_output,
            project_name=project_name,
            dry_run=flow_dry_run,
        )
    except Exception as tool_error:
        deterministic_output = str(tool_error)
    if getattr(browser_task, "output", None) and hasattr(browser_task.output, "raw"):
        browser_task.output.raw = deterministic_output
    return {
        "agent_result": agent_result,
        "agent_output": agent_output,
        "tool_output": deterministic_output,
        "execution_path": "deterministic_tool_fallback",
    }


def check_login(url: str) -> bool:
    auth_file = "auth.json"
    if not os.path.exists(auth_file):
        print(f"⚠️ '{auth_file}' not found.")
        return False

    print("   Launching browser for login check...")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                context = browser.new_context(storage_state=auth_file)
                page = context.new_page()
                page.goto(url, timeout=90000, wait_until="domcontentloaded")
                page.wait_for_timeout(4000)

                title = page.title().lower()
                if "sign in" in title or "login" in title:
                    print(f"   ⚠️ Page title is '{page.title()}' - indicates not logged in.")
                    return False

                print(f"   ✅ Login verified (Page Title: {page.title()})")
                return True
            except Exception as error:
                print(f"   ⚠️ Login check failed: {error}")
                return False
            finally:
                browser.close()
    except Exception as error:
        print(f"   ⚠️ Playwright error: {error}")
        return False


def main() -> None:
    args = parse_cli_args()
    print("## Instagram Agent Crew Initialization ##")

    agents_config = load_config("config/agents.yaml")
    tasks_config = load_config("config/tasks.yaml")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found.")
        print("   Add OPENAI_API_KEY to .env before running this pipeline.")
        return

    cost_mode = get_env_str("COST_MODE", "balanced_saver").lower()
    script_model = get_env_str("OPENAI_MODEL_SCRIPT", "gpt-5.4")
    support_model = get_env_str("OPENAI_MODEL_SUPPORT", "gpt-5.4-mini")
    script_model_fallback = get_env_str(
        "OPENAI_MODEL_SCRIPT_FALLBACK", "gpt-5.4-mini"
    )
    blueprint_model = get_env_str("OPENAI_MODEL_BLUEPRINT", support_model)
    validator_model = get_env_str("OPENAI_MODEL_VALIDATOR", support_model)
    prompt_model = get_env_str("OPENAI_MODEL_PROMPT", support_model)

    if cost_mode == "maximum_savings":
        script_model = script_model_fallback or support_model
        validator_model = support_model
        prompt_model = support_model

    script_temperature = get_env_float("OPENAI_TEMPERATURE_SCRIPT", 0.35)
    support_temperature = get_env_float("OPENAI_TEMPERATURE_SUPPORT", 0.2)
    blueprint_temperature = get_env_float(
        "OPENAI_TEMPERATURE_BLUEPRINT", support_temperature
    )
    validator_temperature = get_env_float("OPENAI_TEMPERATURE_VALIDATOR", 0.15)
    prompt_temperature = get_env_float("OPENAI_TEMPERATURE_PROMPT", 0.2)
    script_reasoning = get_env_reasoning("OPENAI_REASONING_SCRIPT", "medium")
    support_reasoning = get_env_reasoning("OPENAI_REASONING_SUPPORT", "low")
    blueprint_reasoning = get_env_reasoning(
        "OPENAI_REASONING_BLUEPRINT", support_reasoning
    )
    validator_reasoning = get_env_reasoning(
        "OPENAI_REASONING_VALIDATOR", support_reasoning
    )
    prompt_reasoning = get_env_reasoning(
        "OPENAI_REASONING_PROMPT", support_reasoning
    )

    legacy_max_topic_attempts = get_env_int("MAX_TOPIC_ATTEMPTS", 2, minimum=1)
    max_local_repair_attempts = get_env_int(
        "MAX_LOCAL_REPAIR_ATTEMPTS", 2, minimum=1
    )
    max_full_script_rewrites = get_env_int(
        "MAX_FULL_SCRIPT_REWRITES", max(0, legacy_max_topic_attempts - 1), minimum=0
    )
    max_flow_attempts = get_env_int("MAX_FLOW_ATTEMPTS", 1, minimum=1)
    allow_sixth_clip = get_env_bool("ALLOW_SIXTH_CLIP", True)
    retry_on_browser_failure = get_env_bool("RETRY_ON_BROWSER_FAILURE", False)
    auto_fallback_model = get_env_bool("AUTO_FALLBACK_MODEL", True)
    flow_dry_run = get_env_bool("FLOW_DRY_RUN", True)

    if args.flow_live:
        flow_dry_run = False
    if args.flow_dry_run:
        flow_dry_run = True

    script_llm = LLM(
        model=script_model,
        api_key=api_key,
        temperature=script_temperature,
        reasoning_effort=script_reasoning,
    )
    fallback_script_llm = LLM(
        model=script_model_fallback,
        api_key=api_key,
        temperature=script_temperature,
        reasoning_effort=script_reasoning,
    )
    support_llm = LLM(
        model=support_model,
        api_key=api_key,
        temperature=support_temperature,
        reasoning_effort=support_reasoning,
    )
    blueprint_llm = LLM(
        model=blueprint_model,
        api_key=api_key,
        temperature=blueprint_temperature,
        reasoning_effort=blueprint_reasoning,
    )
    validator_llm = LLM(
        model=validator_model,
        api_key=api_key,
        temperature=validator_temperature,
        reasoning_effort=validator_reasoning,
    )
    prompt_llm = LLM(
        model=prompt_model,
        api_key=api_key,
        temperature=prompt_temperature,
        reasoning_effort=prompt_reasoning,
    )

    print(
        f"   🤖 Blueprint model: {blueprint_model} "
        f"(reasoning={blueprint_reasoning}, temp={blueprint_temperature})"
    )
    print(
        f"   🤖 Script model: {script_model} "
        f"(reasoning={script_reasoning}, temp={script_temperature})"
    )
    print(
        f"   🤖 Validator model: {validator_model} "
        f"(reasoning={validator_reasoning}, temp={validator_temperature})"
    )
    print(
        f"   🤖 Prompt model: {prompt_model} "
        f"(reasoning={prompt_reasoning}, temp={prompt_temperature})"
    )
    print(
        f"   🤖 Support model: {support_model} "
        f"(reasoning={support_reasoning}, temp={support_temperature})"
    )
    print(
        "   Cost-safe mode: "
        f"COST_MODE={cost_mode}, "
        f"MAX_LOCAL_REPAIR_ATTEMPTS={max_local_repair_attempts}, "
        f"MAX_FULL_SCRIPT_REWRITES={max_full_script_rewrites}, "
        f"MAX_FLOW_ATTEMPTS={max_flow_attempts}, "
        f"ALLOW_SIXTH_CLIP={allow_sixth_clip}, "
        f"FLOW_DRY_RUN={flow_dry_run}"
    )

    video_tool = VideoGenerationTool()
    video_tool.result_as_answer = True

    story_blueprint_designer = Agent(
        role=agents_config["story_blueprint_designer"]["role"],
        goal=agents_config["story_blueprint_designer"]["goal"],
        backstory=agents_config["story_blueprint_designer"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=blueprint_llm,
    )
    script_writer = Agent(
        role=agents_config["script_writer"]["role"],
        goal=agents_config["script_writer"]["goal"],
        backstory=agents_config["script_writer"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=script_llm,
    )
    prompt_engineer = Agent(
        role=agents_config["prompt_engineer"]["role"],
        goal=agents_config["prompt_engineer"]["goal"],
        backstory=agents_config["prompt_engineer"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=prompt_llm,
    )
    browser_operator = Agent(
        role=agents_config["browser_operator"]["role"],
        goal=agents_config["browser_operator"]["goal"],
        backstory=agents_config["browser_operator"]["backstory"],
        verbose=True,
        allow_delegation=False,
        tools=[video_tool],
        llm=support_llm,
    )
    caption_writer = Agent(
        role=agents_config["caption_writer"]["role"],
        goal=agents_config["caption_writer"]["goal"],
        backstory=agents_config["caption_writer"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=support_llm,
    )
    archivist = Agent(
        role=agents_config["archivist"]["role"],
        goal=agents_config["archivist"]["goal"],
        backstory=agents_config["archivist"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=support_llm,
    )
    script_validator = Agent(
        role=agents_config["script_validator"]["role"],
        goal=agents_config["script_validator"]["goal"],
        backstory=agents_config["script_validator"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=validator_llm,
    )
    editing_advisor = Agent(
        role=agents_config["editing_advisor"]["role"],
        goal=agents_config["editing_advisor"]["goal"],
        backstory=agents_config["editing_advisor"]["backstory"],
        verbose=True,
        allow_delegation=False,
        llm=support_llm,
    )

    topic = None
    original_topic_line = None
    topic_file = "topics.txt"
    completed_file = "completed_topics.txt"
    all_topics: List[str] = []

    if os.path.exists(topic_file):
        with open(topic_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        all_topics = [
            entry.strip()
            for entry in lines
            if entry.strip() and not entry.strip().startswith("#")
        ]
        if all_topics:
            original_topic_line = all_topics[0]
            topic = all_topics[0]
            print(f"\nAuto-selected topic from file: '{topic}'")

    if not topic:
        print("\n(No topics found in 'topics.txt')")
        if sys.stdin and sys.stdin.isatty():
            topic = input("Enter the TOPIC for the Instagram Reel: ")
        else:
            print("No interactive input available. Exiting.")
            print("Action: Add more topics to 'topics.txt'")
            return

    category_override = None
    series_name = None
    series_episode = None
    bracket_pattern = re.compile(r"^\s*(\[[^\]]+\])+")
    bracket_match = bracket_pattern.match(topic)

    if bracket_match:
        tokens = re.findall(r"\[([^\]]+)\]", bracket_match.group(0))
        for token in tokens:
            token_upper = token.upper().strip()
            if token_upper in {
                "COSMOS",
                "MIND",
                "PHYSICS",
                "BIOLOGY",
                "CHEMISTRY",
                "EARTH",
            }:
                category_override = token_upper
                print(f"   🏷️ Category prefix: [{category_override}]")
            elif token_upper.startswith("SERIES:"):
                series_name = token[7:].strip()
                print(f"   🎞️ Series detected: {series_name}")

        topic = re.sub(r"\[[^\]]+\]\s*", "", topic).strip()
        print(f"   📌 Topic (cleaned): '{topic}'")

    if series_name:
        if os.path.exists(completed_file):
            with open(completed_file, "r", encoding="utf-8") as file:
                completed = file.readlines()
            series_episode = (
                sum(1 for entry in completed if series_name.lower() in entry.lower()) + 1
            )
        else:
            series_episode = 1
        print(f"   🎬 Episode number in series: {series_episode}")

    print("\n🔬 Classifying topic...")
    topic_profile_dict = classify_topic(topic, category_override=category_override)
    topic_profile_str = format_profile_for_agent(
        topic_profile_dict,
        series_name=series_name,
        series_episode=series_episode,
    )
    print(f"   ✅ Category detected: [{topic_profile_dict['category']}]")
    print(f"   🎬 Cinematic style: {topic_profile_dict['video_style'][:60]}...")
    print(f"   🎵 Audio: {topic_profile_dict['audio_type'][:60]}...")

    video_url = "https://labs.google/fx/tools/flow"
    print(f"Using Google Flow: {video_url}")

    if flow_dry_run:
        print(
            "\n🧪 Flow dry-run mode enabled. Skipping login/browser checks until local validation passes."
        )
    else:
        print("\n🔎 Checking Login Status...")
        if not check_login(video_url):
            print("\n❌ ABORTING: Login check failed.")
            print("   Please run 'python manual_cookies.py' to update your cookies.")
            return

    task_blueprint = Task(
        description=tasks_config["build_story_blueprint_task"]["description"]
        .replace("{topic}", topic)
        .replace("{topic_profile}", topic_profile_str)
        + f"\n\nALLOW_SIXTH_CLIP: {'true' if allow_sixth_clip else 'false'}",
        expected_output=tasks_config["build_story_blueprint_task"]["expected_output"],
        agent=story_blueprint_designer,
    )
    task_blueprint_base_description = task_blueprint.description

    task1 = Task(
        description=tasks_config["write_script_task"]["description"]
        .replace("{topic}", topic)
        .replace("{topic_profile}", topic_profile_str),
        expected_output=tasks_config["write_script_task"]["expected_output"],
        agent=script_writer,
    )
    task1_base_description = task1.description

    task_validate = Task(
        description=tasks_config["validate_script_task"]["description"]
        + f"\n\nTOPIC PROFILE FOR THIS SCRIPT:\n{topic_profile_str}",
        expected_output=tasks_config["validate_script_task"]["expected_output"],
        agent=script_validator,
        context=[task1],
    )
    task_validate_base_description = task_validate.description

    task2 = Task(
        description=tasks_config["format_json_task"]["description"].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config["format_json_task"]["expected_output"],
        agent=prompt_engineer,
        context=[task_validate],
    )
    task2_base_description = task2.description

    task3 = Task(
        description=tasks_config["generate_video_task"]["description"],
        expected_output=tasks_config["generate_video_task"]["expected_output"],
        agent=browser_operator,
        context=[task2],
        tools=[video_tool],
    )
    task3_base_description = task3.description

    task4 = Task(
        description=tasks_config["generate_caption_task"]["description"].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config["generate_caption_task"]["expected_output"],
        agent=caption_writer,
        context=[task_validate],
    )

    task_editing = Task(
        description=tasks_config["generate_editing_task"]["description"].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config["generate_editing_task"]["expected_output"],
        agent=editing_advisor,
        context=[task_validate, task2, task4],
    )

    archive_dir = os.getenv("ARCHIVE_DIR", "outputs")
    topic_output_dir = os.path.join(archive_dir, safe_project_name(topic))
    os.makedirs(topic_output_dir, exist_ok=True)
    archive_file_path = os.path.join(topic_output_dir, f"{safe_project_name(topic)}.txt")
    print(f"\n📁 Output will be saved to: {topic_output_dir}/")

    task5 = Task(
        description=tasks_config["archive_content_task"]["description"],
        expected_output=tasks_config["archive_content_task"]["expected_output"],
        agent=archivist,
        context=[task_validate, task2, task4, task_editing],
        output_file=archive_file_path,
    )

    blueprint_crew = Crew(
        agents=[story_blueprint_designer],
        tasks=[task_blueprint],
        verbose=True,
        process=Process.sequential,
    )
    script_crew = Crew(
        agents=[script_writer, script_validator],
        tasks=[task1, task_validate],
        verbose=True,
        process=Process.sequential,
    )
    prompt_crew = Crew(
        agents=[prompt_engineer],
        tasks=[task2],
        verbose=True,
        process=Process.sequential,
    )
    browser_crew = Crew(
        agents=[browser_operator],
        tasks=[task3],
        verbose=True,
        process=Process.sequential,
    )
    post_crew = Crew(
        agents=[caption_writer, editing_advisor, archivist],
        tasks=[task4, task_editing, task5],
        verbose=True,
        process=Process.sequential,
    )

    try:
        generation_result: Dict[str, Any] = {}
        script_result = None
        prompt_result = None
        browser_result = None
        post_result = None
        last_video_output = ""
        fail_reason = ""
        execution_success = False
        using_fallback_model = False
        blueprint_retry_guidance = ""
        script_retry_guidance = ""
        prompt_retry_guidance = ""
        validated_blueprint_json = ""
        canonical_validated_script = ""

        total_script_attempts = max_full_script_rewrites + 1
        for rewrite_attempt in range(total_script_attempts):
            task_blueprint.description = (
                f"{task_blueprint_base_description}"
                f"{blueprint_retry_guidance}"
            )

            try:
                blueprint_result = blueprint_crew.kickoff()
            except Exception as blueprint_error:
                fail_reason = f"Blueprint stage failed: {blueprint_error}"
                raise

            blueprint_output = (
                task_blueprint.output.raw
                if getattr(task_blueprint, "output", None)
                and hasattr(task_blueprint.output, "raw")
                and task_blueprint.output.raw
                else str(blueprint_result)
            )

            blueprint_data = {}
            blueprint_errors: List[str]
            try:
                blueprint_data = parse_story_blueprint(blueprint_output)
                blueprint_errors = validate_story_blueprint(
                    blueprint_data, allow_sixth_clip=allow_sixth_clip
                )
            except Exception as blueprint_parse_error:
                blueprint_errors = [f"Blueprint parse error: {blueprint_parse_error}"]

            repair_round = 0
            while blueprint_errors and repair_round < max_local_repair_attempts:
                repair_round += 1
                print(
                    f"🛠️ Running blueprint repair pass {repair_round}/{max_local_repair_attempts}..."
                )
                repair_task = Task(
                    description=(
                        tasks_config["build_story_blueprint_task"]["description"]
                        .replace("{topic}", topic)
                        .replace("{topic_profile}", topic_profile_str)
                        + f"\n\nALLOW_SIXTH_CLIP: {'true' if allow_sixth_clip else 'false'}"
                        + "\n\nCURRENT BLUEPRINT TO FIX:\n"
                        + blueprint_output
                        + "\n\nLOCAL VALIDATION ERRORS TO FIX EXACTLY:\n- "
                        + "\n- ".join(blueprint_errors)
                        + "\n\nCRITICAL: Return ONLY corrected JSON."
                    ),
                    expected_output=tasks_config["build_story_blueprint_task"][
                        "expected_output"
                    ],
                    agent=story_blueprint_designer,
                )
                repair_crew = Crew(
                    agents=[story_blueprint_designer],
                    tasks=[repair_task],
                    verbose=True,
                    process=Process.sequential,
                )
                repair_result = repair_crew.kickoff()
                blueprint_output = (
                    repair_task.output.raw
                    if getattr(repair_task, "output", None)
                    and hasattr(repair_task.output, "raw")
                    and repair_task.output.raw
                    else str(repair_result)
                )
                try:
                    blueprint_data = parse_story_blueprint(blueprint_output)
                    blueprint_errors = validate_story_blueprint(
                        blueprint_data, allow_sixth_clip=allow_sixth_clip
                    )
                except Exception as blueprint_parse_error:
                    blueprint_errors = [
                        f"Blueprint parse error: {blueprint_parse_error}"
                    ]

            if blueprint_errors:
                fail_reason = "Story blueprint failed local validation."
                print(
                    f"❌ Story blueprint failed. Attempt {rewrite_attempt + 1}/{total_script_attempts}."
                )
                for error in blueprint_errors:
                    print(f"   - {error}")
                blueprint_retry_guidance = (
                    f"\n\nCRITICAL RETRY {rewrite_attempt + 1}: "
                    "Return ONLY valid JSON story blueprint. "
                    f"Fix these exact issues: {' | '.join(blueprint_errors)}."
                )
                continue

            validated_blueprint_json = format_canonical_blueprint(blueprint_data)
            generation_result["story_blueprint"] = blueprint_data

            topic_angle = str(blueprint_data.get("topic_angle", topic)).strip() or topic

            task1.description = (
                f"{task1_base_description}"
                f"{script_retry_guidance}\n\n"
                f"TOPIC ANGLE TO USE EXACTLY:\n{topic_angle}\n\n"
                "STORY BLUEPRINT TO FOLLOW EXACTLY:\n"
                f"{validated_blueprint_json}"
            )
            task_validate.description = (
                f"{task_validate_base_description}"
                f"{script_retry_guidance}\n\n"
                f"APPROVED STORY BLUEPRINT:\n{validated_blueprint_json}"
            )

            try:
                script_result = script_crew.kickoff()
            except Exception as kickoff_error:
                error_text = str(kickoff_error)
                model_related_error = (
                    "model" in error_text.lower()
                    or "503" in error_text
                    or "rate limit" in error_text.lower()
                )
                if (
                    model_related_error
                    and not using_fallback_model
                    and script_model_fallback != script_model
                    and auto_fallback_model
                ):
                    print(
                        f"⚠️ Primary script model failed on attempt "
                        f"{rewrite_attempt + 1}/{total_script_attempts}. "
                        f"Switching script writer to fallback model '{script_model_fallback}'."
                    )
                    script_writer.llm = fallback_script_llm
                    using_fallback_model = True
                    continue
                raise

            output_text = (
                task_validate.output.raw
                if getattr(task_validate, "output", None)
                and hasattr(task_validate.output, "raw")
                else ""
            )
            validated_title, validated_clips = extract_script_title_and_clips(output_text)
            local_script_errors = validate_script_clips(validated_clips, blueprint_data)

            repair_round = 0
            while local_script_errors and repair_round < max_local_repair_attempts:
                repair_round += 1
                failed_clips = extract_failed_clips("\n".join(local_script_errors))
                failed_clips_text = (
                    ", ".join(str(clip) for clip in failed_clips)
                    if failed_clips
                    else "unknown"
                )
                print(
                    f"🛠️ Running focused validator repair pass "
                    f"{repair_round}/{max_local_repair_attempts} for clips: {failed_clips_text}..."
                )
                repair_task = Task(
                    description=(
                        tasks_config["validate_script_task"]["description"]
                        + f"\n\nTOPIC PROFILE FOR THIS SCRIPT:\n{topic_profile_str}"
                        + "\n\nAPPROVED STORY BLUEPRINT:\n"
                        + validated_blueprint_json
                        + "\n\nCURRENT SCRIPT TO FIX:\n"
                        + output_text
                        + "\n\nLOCAL VALIDATION ERRORS TO FIX EXACTLY:\n- "
                        + "\n- ".join(local_script_errors)
                        + "\n\nCRITICAL: Rewrite ONLY failing clips if possible. "
                        + "Return ONLY the final corrected script in strict format."
                    ),
                    expected_output=tasks_config["validate_script_task"]["expected_output"],
                    agent=script_validator,
                )
                repair_crew = Crew(
                    agents=[script_validator],
                    tasks=[repair_task],
                    verbose=True,
                    process=Process.sequential,
                )
                repair_result = repair_crew.kickoff()
                output_text = (
                    repair_task.output.raw
                    if getattr(repair_task, "output", None)
                    and hasattr(repair_task.output, "raw")
                    and repair_task.output.raw
                    else str(repair_result)
                )
                validated_title, validated_clips = extract_script_title_and_clips(
                    output_text
                )
                local_script_errors = validate_script_clips(
                    validated_clips, blueprint_data
                )

            if local_script_errors:
                failed_clips = extract_failed_clips("\n".join(local_script_errors))
                failed_clips_text = (
                    ", ".join(str(clip) for clip in failed_clips)
                    if failed_clips
                    else "unknown"
                )
                print(
                    f"❌ Local script validation failed on clips: {failed_clips_text}. "
                    f"Attempt {rewrite_attempt + 1}/{total_script_attempts}."
                )
                for error in local_script_errors:
                    print(f"   - {error}")
                script_retry_guidance = (
                    f"\n\nCRITICAL RETRY {rewrite_attempt + 1}: "
                    f"LOCAL SCRIPT VALIDATION failed on clip(s) {failed_clips_text}. "
                    "Return ONLY the final script in strict format. "
                    f"Fix these exact issues: {' | '.join(local_script_errors)}. "
                    "Do not include analysis or commentary."
                )
                fail_reason = (
                    f"Validated script still violated clarity and continuity rules on clip(s): "
                    f"{failed_clips_text}."
                )
                continue

            canonical_validated_script = format_canonical_script(
                validated_title or topic_angle or topic,
                validated_clips,
            )
            if getattr(task_validate, "output", None) and hasattr(
                task_validate.output, "raw"
            ):
                task_validate.output.raw = canonical_validated_script

            task2.description = (
                f"{task2_base_description}"
                f"{prompt_retry_guidance}\n\n"
                "APPROVED STORY BLUEPRINT:\n"
                f"{validated_blueprint_json}\n\n"
                "VALIDATED SCRIPT TO CONVERT (USE THIS EXACT TEXT FOR voice_text):\n"
                f"{canonical_validated_script}"
            )

            try:
                prompt_result = prompt_crew.kickoff()
            except Exception as prompt_error:
                fail_reason = f"Prompt engineer failed after script validation: {prompt_error}"
                raise

            prompt_json_output = (
                task2.output.raw
                if getattr(task2, "output", None) and hasattr(task2.output, "raw")
                else ""
            )
            if not prompt_json_output:
                print(
                    f"❌ Prompt JSON output missing. Attempt {rewrite_attempt + 1}/{total_script_attempts}."
                )
                prompt_retry_guidance = (
                    f"\n\nCRITICAL RETRY {rewrite_attempt + 1}: "
                    "Return ONLY valid JSON list with 5 or 6 clip objects and all required fields."
                )
                fail_reason = "Prompt engineer returned empty JSON output."
                continue

            generation_result["script_stage"] = script_result
            generation_result["prompt_stage"] = prompt_result

            try:
                write_pipeline_artifacts(
                    topic,
                    blueprint_json=validated_blueprint_json,
                    script_text=canonical_validated_script,
                    prompt_json=prompt_json_output,
                )
            except Exception as artifact_error:
                print(f"⚠️ Could not write pipeline artifacts: {artifact_error}")

            browser_attempt = 0
            while browser_attempt < max_flow_attempts:
                browser_attempt += 1
                browser_stage_result = run_browser_operator_stage(
                    browser_crew=browser_crew,
                    browser_task=task3,
                    browser_task_base_description=task3_base_description,
                    video_tool=video_tool,
                    video_url=video_url,
                    project_name=topic,
                    prompt_json_output=prompt_json_output,
                    flow_dry_run=flow_dry_run,
                )
                browser_result = browser_stage_result.get("agent_result")
                last_video_output = (
                    browser_stage_result.get("tool_output")
                    or browser_stage_result.get("agent_output")
                    or ""
                )
                generation_result["browser_stage"] = browser_stage_result

                print(last_video_output)

                if "PRE-FLIGHT VALIDATION FAILED" in last_video_output:
                    failed_clips = extract_failed_clips(last_video_output)
                    failed_clips_text = (
                        ", ".join(str(clip) for clip in failed_clips)
                        if failed_clips
                        else "unknown"
                    )
                    print(
                        f"❌ Sync/clarity preflight failed on clips: {failed_clips_text}. "
                        f"Attempt {rewrite_attempt + 1}/{total_script_attempts}."
                    )
                    script_retry_guidance = (
                        f"\n\nCRITICAL RETRY {rewrite_attempt + 1}: "
                        f"PRE-FLIGHT failed on clip(s) {failed_clips_text}. "
                        "Keep the sequence smooth, clearer, and more concrete for those failing clips."
                    )
                    prompt_retry_guidance = (
                        f"\n\nCRITICAL RETRY {rewrite_attempt + 1}: "
                        f"PRE-FLIGHT failed on clip(s) {failed_clips_text}. "
                        "Regenerate JSON so visuals stay literal, use the blueprint sync_terms, "
                        "preserve clip_role order, preserve exact voice_text, and avoid invented concepts."
                    )
                    fail_reason = (
                        f"Preflight failed on clip(s): {failed_clips_text}. "
                        "Flow launch was blocked before paid generation."
                    )
                    break

                if "FAILED" in last_video_output.upper() or "❌" in last_video_output:
                    print(
                        f"❌ Browser generation did not complete cleanly. "
                        f"Flow attempt {browser_attempt}/{max_flow_attempts}."
                    )
                    fail_reason = "Browser automation failed before clean generation."
                    if (
                        browser_attempt < max_flow_attempts
                        and retry_on_browser_failure
                        and not flow_dry_run
                    ):
                        print("🔁 Retrying browser stage because paid retry is explicitly enabled.")
                        continue
                    break

                print("✅ Story blueprint + script + sync preflight + browser generation passed.")
                execution_success = True
                break

            if execution_success:
                break

        if not execution_success:
            raise RuntimeError(
                "Pipeline stopped before passing clarity/sync and browser generation gates. "
                f"{fail_reason} "
                "To allow more local rewrites, raise MAX_FULL_SCRIPT_REWRITES. "
                "To allow paid retries, raise MAX_FLOW_ATTEMPTS and enable RETRY_ON_BROWSER_FAILURE."
            )

        print("\n🧾 Running post-production text tasks (caption/edit/archive)...")
        post_result = post_crew.kickoff()

        print("\n\n########################")
        print("## CREW EXECUTION ENDED ##")
        print("########################\n")
        print(generation_result)
        print(post_result)

        if flow_dry_run:
            print(
                f"\n🧪 Dry run complete. Topic '{topic}' remains in the queue because Flow was not launched."
            )
        elif all_topics and all_topics[0] == original_topic_line:
            print(f"\n✅ Marking '{topic}' as complete...")
            remaining_topics = all_topics[1:]
            try:
                with open(topic_file, "w", encoding="utf-8") as file:
                    file.write("\n".join(remaining_topics))
            except Exception as error:
                print(f"⚠️ Error updating topics.txt: {error}")

            try:
                with open(completed_file, "a", encoding="utf-8") as file:
                    file.write(original_topic_line + "\n")
            except Exception as error:
                print(f"⚠️ Error updating completed_topics.txt: {error}")

    except Exception as error:
        print(f"\n❌ CREW FAILED: {error}")
        print(f"   Topic '{topic}' was NOT marked as complete and remains in the queue.")


if __name__ == "__main__":
    main()
