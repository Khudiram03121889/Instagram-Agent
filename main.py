import os
import sys
import json
import argparse

# Force UTF-8 encoding for stdout and stderr to handle emojis on Windows
# This prevents UnicodeEncodeError when printing emojis (like robot faces) to the console or log files.
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions or weird environments where reconfigure might not be available
        pass

import yaml
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tools.browser_tool import VideoGenerationTool
from tools.browser_tool import (
    ALLOWED_BRIDGE_PREFIXES,
    _check_bridge_continuity,
    _sentence_word_counts,
    _word_count,
)
from tools.topic_classifier import classify_topic, format_profile_for_agent
from playwright.sync_api import sync_playwright
import re

# Load environment variables
load_dotenv()

def load_config(file_path):
    # Use utf-8-sig so a UTF-8 BOM does not pollute the first YAML key.
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config format in {file_path}: expected a YAML mapping.")
    return config


def parse_cli_args():
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


def get_env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"⚠️ Invalid float for {name}='{raw}'. Using default {default}.")
        return default


def get_env_reasoning(name, default):
    allowed = {"none", "low", "medium", "high"}
    value = (os.getenv(name) or default).strip().lower()
    if value not in allowed:
        print(f"⚠️ Invalid reasoning_effort '{value}' for {name}. Using '{default}'.")
        return default
    return value


def get_env_int(name, default, minimum=1):
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


def get_env_bool(name, default=False):
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


def extract_failed_clips(text):
    if not text:
        return []
    clips = sorted({int(match) for match in re.findall(r"Clip\s+(\d+)", text)})
    return clips


def extract_script_title_and_clips(text):
    if not text:
        return "", []

    normalized = text.replace("\r\n", "\n")
    title_match = re.search(r"^\s*Title:\s*(.+)$", normalized, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""

    clip_pattern = re.compile(r"^\s*Clip\s+(\d+)\s*:\s*", flags=re.MULTILINE)
    matches = list(clip_pattern.finditer(normalized))
    clips = []

    for idx, match in enumerate(matches):
        clip_number = int(match.group(1))
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        clip_text = normalized[match.end():next_start].strip()
        clip_text = re.sub(r"\s+", " ", clip_text).strip()
        if clip_text:
            clips.append({"clip": clip_number, "voice_text": clip_text})

    return title, clips


def format_canonical_script(title, clips):
    title_line = f"Title: {title or 'Untitled'}"
    clip_lines = [title_line, ""]
    for clip in clips:
        clip_lines.append(f"Clip {clip['clip']}:")
        clip_lines.append(clip["voice_text"])
        clip_lines.append("")
    return "\n".join(clip_lines).strip()


def write_pipeline_artifacts(project_name, script_text=None, prompt_json=None):
    safe_project = "".join(c for c in (project_name or "untitled_project") if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_project:
        safe_project = "untitled_project"
    safe_project = safe_project[:120]
    output_dir = os.path.join("outputs", safe_project)
    os.makedirs(output_dir, exist_ok=True)

    if script_text:
        with open(os.path.join(output_dir, "validated_script.txt"), "w", encoding="utf-8") as f:
            f.write(script_text)

    if prompt_json:
        with open(os.path.join(output_dir, "validated_prompts.json"), "w", encoding="utf-8") as f:
            parsed = json.loads(prompt_json)
            json.dump(parsed, f, ensure_ascii=False, indent=2)


def validate_script_clips(clips):
    errors = []
    if len(clips) != 5:
        errors.append(f"Expected exactly 5 clips, found {len(clips)}.")
        return errors

    expected_numbers = list(range(1, 6))
    actual_numbers = [clip["clip"] for clip in clips]
    if actual_numbers != expected_numbers:
        errors.append(
            f"Expected clip numbering {expected_numbers}, found {actual_numbers}."
        )
        return errors

    for clip in clips:
        clip_number = clip["clip"]
        voice_text = clip["voice_text"]
        words = _word_count(voice_text)
        if words < 15 or words > 17:
            errors.append(f"Clip {clip_number}: word count {words} is outside 15-17.")

        sentence_counts = _sentence_word_counts(voice_text)
        if len(sentence_counts) != 2:
            errors.append(
                f"Clip {clip_number}: must have exactly 2 sentences, found {len(sentence_counts)}."
            )
        else:
            for sentence_idx, sentence_words in enumerate(sentence_counts, start=1):
                if sentence_words < 7 or sentence_words > 9:
                    errors.append(
                        f"Clip {clip_number}: sentence {sentence_idx} has {sentence_words} words (required 7-9)."
                    )

        comma_count = voice_text.count(",")
        if comma_count > 1:
            errors.append(f"Clip {clip_number}: has {comma_count} commas (max 1).")

        if clip_number >= 2 and not any(
            voice_text.lower().startswith(prefix) for prefix in ALLOWED_BRIDGE_PREFIXES
        ):
            errors.append(
                f"Clip {clip_number}: does not start with an approved bridge opener."
            )

    continuity_errors = _check_bridge_continuity(
        [{"voice_text": clip["voice_text"]} for clip in clips]
    )
    errors.extend(continuity_errors)
    return errors

def check_login(url):
    """Checks if auth.json allows a valid login session."""
    auth_file = "auth.json"
    if not os.path.exists(auth_file):
        print(f"⚠️  '{auth_file}' not found.")
        return False
        
    print("   Launching browser for login check...")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True) # Headless is fine for a quick check
            try:
                context = browser.new_context(storage_state=auth_file)
                page = context.new_page()
                page.goto(url, timeout=90000, wait_until="domcontentloaded")
                page.wait_for_timeout(4000)
                
                title = page.title().lower()
                # If title contains 'sign in' or 'login', we are not authenticated
                if "sign in" in title or "login" in title:
                    print(f"   ⚠️  Page title is '{page.title()}' - indicates not logged in.")
                    return False
                
                print("   ✅  Login verified (Page Title: " + page.title() + ")")
                return True
                
            except Exception as e:
                print(f"   ⚠️  Login check failed: {e}")
                return False
            finally:
                browser.close()
    except Exception as e:
        print(f"   ⚠️  Playwright error: {e}")
        return False

def main():
    args = parse_cli_args()
    print("## Instagram Agent Crew Initialization ##")
    
    # Load configurations
    agents_config = load_config('config/agents.yaml')
    tasks_config = load_config('config/tasks.yaml')

    # Validate API key and initialize model routing
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found.")
        print("   Add OPENAI_API_KEY to .env before running this pipeline.")
        return

    script_model = os.getenv("OPENAI_MODEL_SCRIPT", "gpt-5.4")
    support_model = os.getenv("OPENAI_MODEL_SUPPORT", "gpt-5.4-mini")
    script_model_fallback = os.getenv("OPENAI_MODEL_SCRIPT_FALLBACK", "gpt-5.4-mini")
    validator_model = os.getenv("OPENAI_MODEL_VALIDATOR", script_model)
    prompt_model = os.getenv("OPENAI_MODEL_PROMPT", script_model)

    script_temperature = get_env_float("OPENAI_TEMPERATURE_SCRIPT", 0.35)
    support_temperature = get_env_float("OPENAI_TEMPERATURE_SUPPORT", 0.2)
    validator_temperature = get_env_float("OPENAI_TEMPERATURE_VALIDATOR", 0.15)
    prompt_temperature = get_env_float("OPENAI_TEMPERATURE_PROMPT", 0.2)
    script_reasoning = get_env_reasoning("OPENAI_REASONING_SCRIPT", "medium")
    support_reasoning = get_env_reasoning("OPENAI_REASONING_SUPPORT", "low")
    validator_reasoning = get_env_reasoning("OPENAI_REASONING_VALIDATOR", script_reasoning)
    prompt_reasoning = get_env_reasoning("OPENAI_REASONING_PROMPT", script_reasoning)
    max_topic_attempts = get_env_int("MAX_TOPIC_ATTEMPTS", 3, minimum=1)
    retry_on_preflight_failure = get_env_bool("RETRY_ON_PREFLIGHT_FAILURE", True)
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
        reasoning_effort=script_reasoning
    )
    support_llm = LLM(
        model=support_model,
        api_key=api_key,
        temperature=support_temperature,
        reasoning_effort=support_reasoning
    )
    validator_llm = LLM(
        model=validator_model,
        api_key=api_key,
        temperature=validator_temperature,
        reasoning_effort=validator_reasoning
    )
    prompt_llm = LLM(
        model=prompt_model,
        api_key=api_key,
        temperature=prompt_temperature,
        reasoning_effort=prompt_reasoning
    )
    fallback_script_llm = LLM(
        model=script_model_fallback,
        api_key=api_key,
        temperature=script_temperature,
        reasoning_effort=script_reasoning
    )

    print(f"   🤖 Script model: {script_model} (reasoning={script_reasoning}, temp={script_temperature})")
    print(f"   🤖 Validator model: {validator_model} (reasoning={validator_reasoning}, temp={validator_temperature})")
    print(f"   🤖 Prompt model: {prompt_model} (reasoning={prompt_reasoning}, temp={prompt_temperature})")
    print(f"   🤖 Support model: {support_model} (reasoning={support_reasoning}, temp={support_temperature})")
    if script_model_fallback != script_model:
        print(f"   🔁 Script fallback model: {script_model_fallback}")

    print(
        "   Credit-safe mode: "
        f"MAX_TOPIC_ATTEMPTS={max_topic_attempts}, "
        f"RETRY_ON_PREFLIGHT_FAILURE={retry_on_preflight_failure}, "
        f"RETRY_ON_BROWSER_FAILURE={retry_on_browser_failure}, "
        f"FLOW_DRY_RUN={flow_dry_run}"
    )

    # Initialize Tools
    video_tool = VideoGenerationTool()

    # --- Create Agents ---
    script_writer = Agent(
        role=agents_config['script_writer']['role'],
        goal=agents_config['script_writer']['goal'],
        backstory=agents_config['script_writer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=script_llm
    )

    prompt_engineer = Agent(
        role=agents_config['prompt_engineer']['role'],
        goal=agents_config['prompt_engineer']['goal'],
        backstory=agents_config['prompt_engineer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=prompt_llm
    )

    caption_writer = Agent(
        role=agents_config['caption_writer']['role'],
        goal=agents_config['caption_writer']['goal'],
        backstory=agents_config['caption_writer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=support_llm
    )

    archivist = Agent(
        role=agents_config['archivist']['role'],
        goal=agents_config['archivist']['goal'],
        backstory=agents_config['archivist']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=support_llm
    )

    script_validator = Agent(
        role=agents_config['script_validator']['role'],
        goal=agents_config['script_validator']['goal'],
        backstory=agents_config['script_validator']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=validator_llm
    )

    editing_advisor = Agent(
        role=agents_config['editing_advisor']['role'],
        goal=agents_config['editing_advisor']['goal'],
        backstory=agents_config['editing_advisor']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=support_llm
    )

    # --- Get Topic (Auto or Manual) ---
    topic = None
    original_topic_line = None  # Safety default for manual input mode
    topic_file = "topics.txt"
    completed_file = "completed_topics.txt"
    all_topics = []
    
    # Try to read from file first
    if os.path.exists(topic_file):
        with open(topic_file, 'r') as f:
            lines = f.readlines()
        
        # Filter out empty lines and comments
        all_topics = [t.strip() for t in lines if t.strip() and not t.strip().startswith("#")]
        
        if all_topics:
            original_topic_line = all_topics[0]
            topic = all_topics[0]
            print(f"\nAuto-selected topic from file: '{topic}'")
    
    if not topic:
        print("\n(No topics found in 'topics.txt')")
        # Check if running interactively (terminal)
        import sys
        if sys.stdin and sys.stdin.isatty():
            topic = input("Enter the TOPIC for the Instagram Reel: ")
        else:
            print("No interactive input available. Exiting.")
            print("Action: Add more topics to 'topics.txt'")
            return
    
    # --- Parse Category and Series Prefix ---
    # Supports formats like:
    # [MIND] Topic name
    # [MIND][SERIES:BrainLies] Topic name
    # [COSMOS][SERIES:InvisibleUniverse] Topic name

    category_override = None
    series_name = None
    series_episode = None

    # Extract all bracketed tokens from the start of the topic
    bracket_pattern = re.compile(r'^\s*(\[[^\]]+\])+')
    bracket_match = bracket_pattern.match(topic)

    if bracket_match:
        # Find all individual bracket tokens
        tokens = re.findall(r'\[([^\]]+)\]', bracket_match.group(0))
        
        for token in tokens:
            token_upper = token.upper().strip()
            
            # Check if it is a category token
            if token_upper in ["COSMOS", "MIND", "PHYSICS", 
                               "BIOLOGY", "CHEMISTRY", "EARTH"]:
                category_override = token_upper
                print(f"   🏷️  Category prefix: [{category_override}]")
            
            # Check if it is a series token
            elif token_upper.startswith("SERIES:"):
                series_name = token[7:].strip()  # Remove "SERIES:" prefix
                print(f"   🎞️  Series detected: {series_name}")
        
        # Clean the topic — remove all bracket tokens
        topic = re.sub(r'\[[^\]]+\]\s*', '', topic).strip()
        print(f"   📌 Topic (cleaned): '{topic}'")

    # --- Count episode number if series detected ---
    if series_name:
        if os.path.exists(completed_file):
            with open(completed_file, 'r') as f:
                completed = f.readlines()
            # Count how many completed topics had the same series name
            series_episode = sum(
                1 for t in completed 
                if series_name.lower() in t.lower()
            ) + 1
        else:
            series_episode = 1  # No completed file yet — this is Episode 1
        print(f"   🎬 Episode number in series: {series_episode}")

    # --- Classify Topic & Build Profile ---
    print(f"\n🔬 Classifying topic...")
    topic_profile_dict = classify_topic(topic, category_override=category_override)
    topic_profile_str = format_profile_for_agent(
        topic_profile_dict,
        series_name=series_name,
        series_episode=series_episode
    )
    print(f"   ✅ Category detected: [{topic_profile_dict['category']}]")
    print(f"   🎬 Cinematic style: {topic_profile_dict['video_style'][:60]}...")
    print(f"   🎵 Audio: {topic_profile_dict['audio_type'][:60]}...")

    # Google Flow URL
    video_url = "https://labs.google/fx/tools/flow"
    print(f"Using Google Flow: {video_url}")

    # --- Pre-flight Check: Validate Login ---
    if flow_dry_run:
        print("\n🧪 Flow dry-run mode enabled. Skipping login/browser checks until local validation passes.")
    else:
        print("\n🔎 Checking Login Status...")
        if not check_login(video_url):
            print("\n❌ ABORTING: Login check failed.")
            print("   Please run 'python manual_cookies.py' to update your cookies.")
            return

    # --- Create Tasks ---
    
    # Task 1: Script
    task1 = Task(
        description=tasks_config['write_script_task']['description'].replace(
            "{topic}", topic
        ).replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config['write_script_task']['expected_output'],
        agent=script_writer
    )

    # Task 1.5: Script Validation
    task_validate = Task(
        description=tasks_config['validate_script_task']['description'] + 
        f"\n\nTOPIC PROFILE FOR THIS SCRIPT:\n{topic_profile_str}",
        expected_output=tasks_config['validate_script_task']['expected_output'],
        agent=script_validator,
        context=[task1]
    )

    # Task 2: JSON Prompts (now receives VALIDATED script)
    task2 = Task(
        description=tasks_config['format_json_task']['description'].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config['format_json_task']['expected_output'],
        agent=prompt_engineer,
        context=[task_validate]
    )
    task2_base_description = task2.description

    # Video generation is executed directly via video_tool._run()
    # after script + validation + prompt tasks finish successfully.

    # Task 4: Caption Generation
    task4 = Task(
        description=tasks_config['generate_caption_task']['description'].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config['generate_caption_task']['expected_output'],
        agent=caption_writer,
        context=[task_validate]  # Needs validated script
    )

    # Task 4.5: Editing Instructions
    task_editing = Task(
        description=tasks_config['generate_editing_task']['description'].replace(
            "{topic_profile}", topic_profile_str
        ),
        expected_output=tasks_config['generate_editing_task']['expected_output'],
        agent=editing_advisor,
        context=[task_validate, task2, task4]  # needs validated script + JSON prompts + caption
    )

    # Task 5: Archiving
    task5_desc = tasks_config['archive_content_task']['description'].replace("{topic}", topic)
    
    # Generate safe filename and directory for task 5 output
    safe_topic = "".join(c for c in topic if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_topic:
        safe_topic = "untitled_project"
    safe_topic = safe_topic[:120]
    
    # Use 'outputs/' as the root folder (inside the agent project folder)
    # Each topic gets its own subfolder: outputs/<topic_name>/
    archive_dir = os.getenv("ARCHIVE_DIR", "outputs")
    topic_output_dir = os.path.join(archive_dir, safe_topic)
    os.makedirs(topic_output_dir, exist_ok=True)
    archive_file_path = os.path.join(topic_output_dir, f"{safe_topic}.txt")
    
    print(f"\n📁 Output will be saved to: {topic_output_dir}/")
    
    task5 = Task(
        description=task5_desc,
        expected_output=tasks_config['archive_content_task']['expected_output'],
        agent=archivist,
        context=[task_validate, task2, task4, task_editing],
        output_file=archive_file_path
    )

    BANNED = ["vast", "void", "adrift", "ignites", "default network",
              "phenomenon", "resonance", "cosmic whisper", "immense",
              "ethereal", "cerebral", "imperceptible", "visceral"]
    
    def check_banned_words(script_text):
        found = [w for w in BANNED if w.lower() in script_text.lower()]
        if found:
            print(f"⚠️ BANNED WORDS FOUND: {found}")
            return False
        return True

    # --- Create Split Crews ---
    script_crew = Crew(
        agents=[script_writer, script_validator],
        tasks=[task1, task_validate],
        verbose=True,
        process=Process.sequential
    )

    prompt_crew = Crew(
        agents=[prompt_engineer],
        tasks=[task2],
        verbose=True,
        process=Process.sequential
    )

    post_crew = Crew(
        agents=[caption_writer, editing_advisor, archivist],
        tasks=[task4, task_editing, task5],
        verbose=True,
        process=Process.sequential
    )

    # --- Kickoff ---
    try:
        using_fallback_model = False
        generation_result = None
        script_result = None
        prompt_result = None
        post_result = None
        execution_success = False
        last_video_output = ""
        fail_reason = ""
        task2_retry_guidance = ""

        for attempt in range(max_topic_attempts):
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
                        f"⚠️ Primary script model failed on attempt {attempt+1}/{max_topic_attempts}. "
                        f"Switching script writer to fallback model '{script_model_fallback}'."
                    )
                    script_writer.llm = fallback_script_llm
                    using_fallback_model = True
                    continue
                raise

            output_text = task_validate.output.raw if getattr(task_validate, 'output', None) and hasattr(task_validate.output, 'raw') else ""
            validated_title, validated_clips = extract_script_title_and_clips(output_text)
            local_script_errors = validate_script_clips(validated_clips)

            if local_script_errors:
                failed_clips = extract_failed_clips("\n".join(local_script_errors))
                failed_clips_text = ", ".join(str(c) for c in failed_clips) if failed_clips else "unknown"
                print(
                    f"❌ Local script validation failed on clips: {failed_clips_text}. "
                    f"Attempt {attempt+1}/{max_topic_attempts}."
                )
                for error in local_script_errors:
                    print(f"   - {error}")
                task_validate.description += (
                    f"\n\nCRITICAL RETRY {attempt+1}: LOCAL SCRIPT VALIDATION failed on "
                    f"clip(s) {failed_clips_text}. Return ONLY the final 5-clip script in strict format. "
                    f"Fix these exact issues: {' | '.join(local_script_errors)}. "
                    "Do not include analysis or commentary."
                )
                fail_reason = (
                    f"Validated script still violated timing/continuity rules on clip(s): "
                    f"{failed_clips_text}."
                )
                continue

            canonical_validated_script = format_canonical_script(
                validated_title or topic,
                validated_clips
            )
            if getattr(task_validate, 'output', None) and hasattr(task_validate.output, 'raw'):
                task_validate.output.raw = canonical_validated_script
            output_text = canonical_validated_script

            task2.description = (
                f"{task2_base_description}"
                f"{task2_retry_guidance}\n\n"
                "VALIDATED SCRIPT TO CONVERT (USE THIS EXACT TEXT FOR voice_text):\n"
                f"{canonical_validated_script}"
            )

            try:
                prompt_result = prompt_crew.kickoff()
            except Exception as prompt_error:
                fail_reason = f"Prompt engineer failed after script validation: {prompt_error}"
                raise

            generation_result = {
                "script_stage": script_result,
                "prompt_stage": prompt_result,
            }
            prompt_json_output = task2.output.raw if getattr(task2, 'output', None) and hasattr(task2.output, 'raw') else ""
            try:
                write_pipeline_artifacts(
                    topic,
                    script_text=canonical_validated_script,
                    prompt_json=prompt_json_output if prompt_json_output else None
                )
            except Exception as artifact_error:
                print(f"⚠️ Could not write pipeline artifacts: {artifact_error}")

            if not check_banned_words(output_text):
                print(f"❌ Banned words found. Attempt {attempt+1}/{max_topic_attempts}.")
                task1.description += (
                    f"\n\nCRITICAL RETRY {attempt+1}: You used BANNED WORDS. "
                    "NEVER USE: vast, void, adrift, ignites, default network, phenomenon, "
                    "resonance, cosmic whisper, immense, ethereal, cerebral, imperceptible, visceral."
                )
                fail_reason = "Script validation failed due banned words."
                continue

            if not prompt_json_output:
                print(f"❌ Prompt JSON output missing. Attempt {attempt+1}/{max_topic_attempts}.")
                task2.description += (
                    f"\n\nCRITICAL RETRY {attempt+1}: Return ONLY valid JSON list with all required fields."
                )
                fail_reason = "Prompt engineer returned empty JSON output."
                continue

            print("⚙️ Running browser video generation tool directly...")
            try:
                last_video_output = video_tool._run(
                    url=video_url,
                    json_content=prompt_json_output,
                    project_name=topic,
                    dry_run=flow_dry_run
                )
            except Exception as direct_video_error:
                last_video_output = str(direct_video_error)

            print(last_video_output)

            if "PRE-FLIGHT VALIDATION FAILED" in last_video_output:
                failed_clips = extract_failed_clips(last_video_output)
                failed_clips_text = ", ".join(str(c) for c in failed_clips) if failed_clips else "unknown"
                print(
                    f"❌ Sync/timing preflight failed on clips: {failed_clips_text}. "
                    f"Attempt {attempt+1}/{max_topic_attempts}."
                )
                task_validate.description += (
                    f"\n\nCRITICAL RETRY {attempt+1}: PRE-FLIGHT failed. "
                    f"Fix ONLY failing clips ({failed_clips_text}) for timing choreography and bridge continuity. "
                    "Do NOT rewrite passing clips."
                )
                task2_retry_guidance += (
                    f"\n\nCRITICAL RETRY {attempt+1}: PRE-FLIGHT failed on clips ({failed_clips_text}). "
                    "For those clips only, regenerate JSON to satisfy: "
                    "word_target=16, words 15-17, estimated_sec 7.6-8.2, implied speech speed 1.95-2.10, "
                    "exact beat_map timings, motion_intensity='fast_structured', and strong visual/script overlap."
                )
                fail_reason = (
                    f"Preflight failed on clip(s): {failed_clips_text}. "
                    "Regenerating upstream content is safe because Flow was not triggered yet."
                )
                if not retry_on_preflight_failure:
                    print(
                        "Credit-safe stop: retry on preflight failure is disabled. "
                        "Set RETRY_ON_PREFLIGHT_FAILURE=true to auto-regenerate."
                    )
                    break
                continue

            if "FAILED" in last_video_output.upper() or "❌" in last_video_output:
                print(
                    f"❌ Browser generation did not complete cleanly. "
                    f"Attempt {attempt+1}/{max_topic_attempts}."
                )
                fail_reason = "Browser automation failed before clean generation."
                if not retry_on_browser_failure:
                    print(
                        "Credit-safe stop: retry on browser failure is disabled. "
                        "Set RETRY_ON_BROWSER_FAILURE=true to retry automatically."
                    )
                    break
                continue

            print("✅ Script + preflight + browser generation passed.")
            execution_success = True
            break

        if not execution_success:
            raise RuntimeError(
                "Pipeline stopped before passing timing/sync and browser generation gates. "
                f"{fail_reason} "
                "To allow paid retries, raise MAX_TOPIC_ATTEMPTS and enable "
                "RETRY_ON_PREFLIGHT_FAILURE/RETRY_ON_BROWSER_FAILURE."
            )

        print("\n🧾 Running post-production text tasks (caption/edit/archive)...")
        post_result = post_crew.kickoff()

        print("\n\n########################")
        print("## CREW EXECUTION ENDED ##")
        print("########################\n")
        print(generation_result)
        print(post_result)

        # --- Success! Update Files ---
        # Only now do we move the topic to completed
        if flow_dry_run:
            print(f"\n🧪 Dry run complete. Topic '{topic}' remains in the queue because Flow was not launched.")
        elif all_topics and all_topics[0] == original_topic_line:
            print(f"\n✅ Marking '{topic}' as complete...")
            
            # Remove from topics.txt
            remaining_topics = all_topics[1:]
            try:
                with open(topic_file, 'w') as f:
                    f.write("\n".join(remaining_topics))
            except Exception as e:
                print(f"⚠️ Error updating topics.txt: {e}")

            # Add to completed_topics.txt
            try:
                with open(completed_file, 'a') as f:
                    f.write(original_topic_line + "\n")
            except Exception as e:
                print(f"⚠️ Error updating completed_topics.txt: {e}")
                
    except Exception as e:
        print(f"\n❌ CREW FAILED: {e}")
        print(f"   Topic '{topic}' was NOT marked as complete and remains in the queue.")

if __name__ == "__main__":
    main()
