# Instagram Science Reel Generation: Master Scheduled Task Instructions

This document contains the exact instruction payload to set up a daily scheduled task in Antigravity. When triggered, it will run the end-to-end multi-stage pipeline, pausing at the **Script Approval Gate** before automating Google Flow.

---

## 📅 Scheduling in Antigravity

To schedule this task to run daily at a specific time (e.g., every day at 10:00 AM), you can type the `/schedule` slash command in the chat, or run a schedule definition like this:

```markdown
/schedule CronExpression="0 10 * * *" MaxIterations="unlimited" Prompt="Run the Daily Instagram Reel Generation Pipeline"
```

---

## 📋 The Scheduled Task Instruction Payload
*Copy and paste the entire block below as the instruction/prompt for the scheduled task:*

```markdown
You are executing the "Daily Instagram Reel Generation Pipeline" for the Instagram Science Reel project. Perform the following steps sequentially, strictly adhering to the contract validation rules and the user approval gate.

### STAGE 1: TOPIC EXTRACTION
1. Read the first line of "topics.txt" located in the project root. This is today's topic.
2. If "topics.txt" is empty, halt and notify the user.
3. Clean the topic text and determine its science category using "tools/topic_classifier.py" (or matching COSMOS, MIND, PHYSICS, BIOLOGY, CHEMISTRY, or EARTH rules).

### STAGE 2: STORY BLUEPRINT DESIGN (5-CLIP MODE)
Generate a story blueprint in JSON format under "outputs/temp_story_blueprint.json". The blueprint must strictly enforce the following modes and target durations:
- Clip 1: 6 seconds (Role: Hook / Eye-catching science phenomenon)
- Clip 2: 8 seconds (Role: The core science mechanism/cause)
- Clip 3: 8 seconds (Role: Deep-dive/supporting biological or physical command)
- Clip 4: 8 seconds (Role: The visual/pressure safety shield explanation)
- Clip 5: 4 seconds (Role: Satisfying takeaway/outroduction)

Each clip must include fields: `clip_number`, `clip_role`, `core_idea`, `bridge_from_previous`, `next_clip_seed`, `viewer_takeaway`, `visual_anchor_terms`, and `duration_seconds`.

### STAGE 3: SCRIPT WRITING & WORD LIMITS
Write a premium educational narration script based on the blueprint:
1. **STRICT Word Count Limits** (Failing these limits will truncate the narration mid-speech):
   - 6-second clip: 12 to 14 words max.
   - 8-second clip: 16 to 24 words max (FAIL at 25+ words).
   - 4-second clip: 8 to 12 words max.
2. **Style Rules**:
   - Sound educational, direct, and conversational.
   - **Banned Words**: Never use vague, dramatic, or poetic documentary filler (e.g., "ever wondered", "imagine", "delve", "explore", "mysterious", "dance of", "symphony").
   - Seamless bridging: Clips 2+ must echo ideas or terms from the preceding clip to maintain narrative flow.

### STAGE 4: 🛑 USER APPROVAL GATE (CRITICAL STOP)
Stop execution here. Display the final written script (including word counts for each clip) and the story blueprint clearly to the user.
Wait and ask: "Please review and approve this script to launch Google Flow video generation."
Do NOT proceed to Stage 5, do NOT call any browser tools, and do NOT modify any other files until the user explicitly responds with approval (e.g., "Approved", "Go", "Yes").

### STAGE 5: OMNI-PROMPT GENERATION
Once approved, convert the script into visual video generation prompts:
1. Generate descriptive cinematic prompts following the visual-first guidelines.
2. **STRICT Visual Prompt Rules**: No graphic text, labels, numbers, letters, titles, subtitles, or text overlays should be described in the prompt to prevent letters from rendering in the video frames.
3. Save the finalized prompts to "outputs/<topic>/validated_prompts.json" in JSON format.

### STAGE 6: GOOGLE FLOW AUTOMATION (VEO OMNI FLASH)
Automate Google Flow to generate the clips:
1. Run "python scratch/extract_profile_cookies.py" to extract active login session cookies from Chrome and write them to "auth.json".
2. Run "python -u scratch/generate_reels.py" in the project workspace directory.
3. This automated script will:
   - Load cookies from "auth.json" and launch a Playwright browser.
   - Verify general settings: Video Mode, Portrait Mode, x2 duration parameter.
   - Verify and switch the model to "Omni Flash" using direct DOM evaluations to bypass stale-element limits.
   - Dynamic Clip Settings: Before submitting the prompt for *each* clip, the script opens the settings menu, checks if the target duration (e.g. 6s, 8s, or 4s) is selected, toggles the correct duration option if needed, closes the menu, and then enters the prompt and generates.
   - Monitor each clip generation until successfully completed.

### STAGE 7: PREMIUM CAPTIONS GENERATION & DATE STORAGE
1. Generate two captions for the final reel:
   - **Instagram Caption**: Long-form, engaging, formatted with emojis, detailed scientific explanation blocks, a clear CTA, and relevant science hashtags.
   - **YouTube Shorts Caption**: Strictly under 100 characters, punchy, with high-reach hashtags.
2. Retrieve the current date in YYYY-MM-DD format (e.g., "2026-06-01").
3. Create a subfolder "outputs/<YYYY-MM-DD>/<topic-name>/" and save:
   - "validated_script.txt"
   - "validated_prompts.json"
   - "captions.txt" (containing both Instagram and YouTube captions).
4. Save a duplicate copy of "captions.txt" in the root of the date directory: "outputs/<YYYY-MM-DD>/captions.txt".

### STAGE 8: QUEUE MAINTENANCE
1. Move the processed topic from the first line of "topics.txt" to the end of "completed_topics.txt".
2. Rewrite "topics.txt" to remove the processed topic, shifting the remaining queue up.
3. Print a final summary of the completed reel files and links.
```
