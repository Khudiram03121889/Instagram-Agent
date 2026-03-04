import os
import sys

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
from tools.file_tool import FileArchiverTool
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
                page.goto(url, timeout=30000)
                page.wait_for_load_state("networkidle")
                
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
    print("## Instagram Agent Crew Initialization ##")
    
    # Load configurations
    agents_config = load_config('config/agents.yaml')
    tasks_config = load_config('config/tasks.yaml')
    
    # Initialize LLM
    my_llm = LLM(
        model="gemini/gemini-2.5-flash-lite",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7
    )

    # Initialize Tools
    video_tool = VideoGenerationTool()
    file_tool = FileArchiverTool()

    # --- Create Agents ---
    script_writer = Agent(
        role=agents_config['script_writer']['role'],
        goal=agents_config['script_writer']['goal'],
        backstory=agents_config['script_writer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=my_llm
    )

    prompt_engineer = Agent(
        role=agents_config['prompt_engineer']['role'],
        goal=agents_config['prompt_engineer']['goal'],
        backstory=agents_config['prompt_engineer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=my_llm
    )

    browser_operator = Agent(
        role=agents_config['browser_operator']['role'],
        goal=agents_config['browser_operator']['goal'],
        backstory=agents_config['browser_operator']['backstory'],
        verbose=True,
        allow_delegation=False,
        tools=[video_tool],
        llm=my_llm
    )

    caption_writer = Agent(
        role=agents_config['caption_writer']['role'],
        goal=agents_config['caption_writer']['goal'],
        backstory=agents_config['caption_writer']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=my_llm
    )

    archivist = Agent(
        role=agents_config['archivist']['role'],
        goal=agents_config['archivist']['goal'],
        backstory=agents_config['archivist']['backstory'],
        verbose=True,
        allow_delegation=False,
        tools=[file_tool],
        llm=my_llm
    )

    script_validator = Agent(
        role=agents_config['script_validator']['role'],
        goal=agents_config['script_validator']['goal'],
        backstory=agents_config['script_validator']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=my_llm
    )

    editing_advisor = Agent(
        role=agents_config['editing_advisor']['role'],
        goal=agents_config['editing_advisor']['goal'],
        backstory=agents_config['editing_advisor']['backstory'],
        verbose=True,
        allow_delegation=False,
        llm=my_llm
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
        description=tasks_config['validate_script_task']['description'],
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

    # Task 3: Video Generation
    task3_desc = tasks_config['generate_video_task']['description']
    task3_desc += f"\n\nTARGET URL: {video_url}"
    task3_desc += f"\nPROJECT NAME: {topic}"
    
    task3 = Task(
        description=task3_desc,
        expected_output=tasks_config['generate_video_task']['expected_output'],
        agent=browser_operator,
        context=[task2],
        tools=[video_tool]
    )

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
        context=[task_validate, task4]  # needs validated script + caption
    )

    # Task 5: Archiving
    task5_desc = tasks_config['archive_content_task']['description'].replace("{topic}", topic)
    
    task5 = Task(
        description=task5_desc,
        expected_output=tasks_config['archive_content_task']['expected_output'],
        agent=archivist,
        context=[task_validate, task2, task4, task_editing],
        tools=[file_tool]
    )

    # --- Create Crew ---
    crew = Crew(
        agents=[script_writer, script_validator, prompt_engineer, browser_operator, caption_writer, editing_advisor, archivist],
        tasks=[task1, task_validate, task2, task3, task4, task_editing, task5],
        verbose=True,
        process=Process.sequential
    )

    # --- Kickoff ---
    try:
        result = crew.kickoff()
        
        print("\n\n########################")
        print("## CREW EXECUTION ENDED ##")
        print("########################\n")
        print(result)

        # --- Success! Update Files ---
        # Only now do we move the topic to completed
        if all_topics and all_topics[0] == original_topic_line:
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
