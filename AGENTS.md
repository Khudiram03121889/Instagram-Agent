# AGENTS.md

This file provides guidance to Antigravity when working with code in this repository.

## Project Overview

This is an **Instagram Science Reel Generator** — an Antigravity-native pipeline that creates short-form educational video content (40-50 seconds, 5-6 clips) for Instagram Reels using Google Flow with **Omni Flash**.

**The pipeline does NOT use CrewAI agents.** All orchestration is done directly by Antigravity.

## Pipeline Architecture (4 Phases)

```
Phase 1: Blueprint + Script → HUMAN REVIEW GATE
Phase 2: Prompt Generation (after approval)
Phase 3: Browser Automation (Google Flow via /browser)
Phase 4: Captions + Archive
```

### Phase 1: Blueprint + Script Generation

1. **Read the first uncommented topic** from `topics.txt`
2. **Classify the topic** using `tools/topic_classifier.py`:
   - Import and call `classify_topic(topic)` to get the topic profile
   - This returns: category, video_style, audio_type, sfx_layers, voice_profile, etc.
   - Category override: topics prefixed with `[COSMOS]`, `[MIND]`, etc. force a category
3. **Generate a story blueprint** (JSON) with these rules:
   - 5 clips (default) or 6 clips if topic needs more room
   - Required fields per clip: `clip_number`, `clip_role`, `core_idea`, `bridge_from_previous`, `next_clip_seed`, `viewer_takeaway`, `visual_anchor_terms`, `hook_pattern`, `retention_reason`, `visual_premise`, `camera_plan`, `duration_seconds`, `viewer_emotion`
   - Role order for 5-clip: Hook → Question → Mechanism → Contrast/Payoff → Personal Takeaway
   - Role order for 6-clip: Hook → Question → Mechanism Part 1 → Mechanism Part 2 → Contrast/Payoff → Personal Takeaway
   - `duration_seconds` must be one of: 4, 6, 8, 10
   - Total reel: 24-36 seconds
   - `visual_anchor_terms`: 2-4 literal nouns/actions (no abstract metaphors)
4. **Generate a script** following the blueprint:
   - Write one continuous Full Script first, then divide into clips
   - Word limits (STRICT): 4s = 10 words, 6s = 15 words, 8s = 20 words, 10s = 25 words
   - Simple English a 14-year-old can follow
   - Clips 2+ must bridge from the previous clip
   - No jargon, no vague lines, no mystery-box phrasing
5. **Validate the script** against `pipeline_contract.py` constants
6. **STOP AND PRESENT TO USER** — Show:
   - The blueprint (clip roles, durations, visual anchors)
   - The full script with per-clip text
   - Total duration and word counts per clip
   - Ask: "Does this look good? Say 'proceed' to generate video prompts."

> **CRITICAL**: Do NOT proceed to Phase 2 without explicit user approval.

### Phase 2: Prompt Generation

After user approval, generate Omni Flash prompts for each clip:

1. **Read `gemini-omni-prompt-generator.skill`** for formatting rules
2. **For each clip**, create a prompt object with:
   - `clip_label`: "CLIP 1", "CLIP 2", etc.
   - `clip_role`: From blueprint
   - `duration_seconds`: From blueprint (the /browser agent will select this duration tab)
   - `voice_text`: Exact script text (never rewrite)
   - `sync_terms`: Visual elements that must appear in both narration AND visual
   - `visual_goal`: What the viewer should understand
   - `voice`: Identical across ALL clips (from topic_classifier's LOCKED_VOICE_PROFILE)
   - `background_audio`: Identical across ALL clips (from topic profile)
   - `visual`: Formatted as:
     ```
     [Camera]: [SHOT TYPE], [CAMERA MOTION]
     [Style]: [AESTHETIC], [TEXTURE/FEEL]
     [Lighting]: [LIGHT SOURCE], [LIGHT QUALITY]
     [Location]: [ENVIRONMENT DESCRIPTION]
     [Action]: [CHARACTERS/OBJECTS + ACTIONS matching sync_terms]
     [Extra]: A {voice_tone} {voice_gender} narrator says: "{voice_text}". Speed: 1.02. Ambient sound bed: {sfx_layers}. Portrait 9:16. No on-screen text.
     ```
   - `video_style`: From topic profile
   - `orientation`: "portrait"
   - `aspect_ratio`: "9:16"
3. **Pre-flight validation**:
   - All clips must have identical voice settings
   - All clips must have identical background_audio settings
   - sync_terms must appear in both voice_text AND visual
   - voice.speed must be >= 1.0
   - No on-screen text/subtitles/labels in visual prompts
4. **Save outputs**:
   - `outputs/<topic-name>/story_blueprint.json`
   - `outputs/<topic-name>/validated_script.txt`
   - `outputs/<topic-name>/validated_prompts.json`

### Phase 3: Browser Automation (Google Flow)

Use the **`/browser` agent** to generate videos on Google Flow.

#### Pre-requisite: Cookie Authentication
- `auth.json` must exist (created by `python manual_cookies.py`)
- If `auth.json` is missing or expired, ask the user to run `python manual_cookies.py`

#### Google Flow Settings (EXACT — match these precisely)
| Setting | Value | How to Select |
|---------|-------|---------------|
| **Mode** | Video | Click the "Video" tab (not Image) |
| **Sub-mode** | Ingredients | Click the "Ingredients" tab (not Frames) |
| **Orientation** | 9:16 | Click the "9:16" tab (not 16:9) |
| **Variations** | x2 | Click the "x2" tab |
| **Model** | Omni Flash | Click the model dropdown → select "Omni Flash" |
| **Duration** | Per-clip | Switch the duration tab (4s/6s/8s/10s) BEFORE each clip |

#### Generation Sequence
1. Navigate to `https://labs.google/fx/tools/flow`
2. Dismiss any changelog/welcome modals
3. Set ALL settings above (Video, Ingredients, 9:16, x2, Omni Flash)
4. **For Clip 1**:
   - Select the correct duration tab (e.g., 4s for a Hook clip)
   - Enter the visual prompt from `validated_prompts.json` (the `visual` field)
   - Click Generate (or Ctrl+Enter)
   - Wait for generation to complete (2-3 minutes)
5. **For Clips 2+**:
   - Switch the duration tab to match this clip's `duration_seconds`
   - Click the "Extend" button to maintain visual continuity
   - Enter the visual prompt
   - Click Generate
   - Wait for completion
6. Rename the project to the topic name

### Phase 4: Captions + Archive

Generate two separate captions:

#### YouTube Title (≤100 characters)
- Short, curiosity-driven, no hashtags
- Must include the core hook from the reel
- Example: `"Why nothing you touch is actually solid 🔬"`

#### Instagram Caption (unlimited, rich)
- Opening hook line matching the reel's first clip
- 2-3 lines of reflective commentary
- Call-to-action (save/share)
- 10-15 relevant hashtags
- 2-4 emojis sprinkled naturally

#### Archive
- Save captions to `outputs/<topic-name>/captions.txt`
- Combine script + prompts + captions into `outputs/<topic-name>/<topic-name>.txt`

### Topic Completion

After ALL phases complete successfully:
1. **Remove the topic** from `topics.txt` (delete the first line)
2. **Append the topic** to `completed_topics.txt`
3. Confirm to the user: "✅ Topic '<name>' marked as complete."

## Validation Rules (from pipeline_contract.py)

### Word Limits (STRICT)
- 4-second clip: max 10 words
- 6-second clip: max 15 words
- 8-second clip: max 20 words
- 10-second clip: max 25 words
- Words per second: 2.5

### Blueprint Validation
- Clip count: 5 or 6 only
- Role sequence must follow the exact order
- All required fields must be present
- `duration_seconds` must be in {4, 6, 8, 10}
- Total duration: 24-36 seconds
- `visual_anchor_terms`: 2-4 literal nouns (no metaphors)

### Script Validation
- Each clip must not exceed its word limit
- Clips 2+ must bridge from previous clip concepts
- No banned words/phrases (documentary-style, vague filler)
- Full Script = concatenation of all clip texts (division rule)

### Prompt JSON Validation
- Voice settings identical across all clips
- Background audio identical across all clips
- sync_terms must appear in both voice_text AND visual
- voice.speed >= 1.0
- No on-screen text in visual prompts

## Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | This file — master instructions for Antigravity |
| `INSTRUCTIONS.md` | Detailed validation rules and examples |
| `topics.txt` | Queue of topics to process |
| `completed_topics.txt` | Completed topic archive |
| `pipeline_contract.py` | Validation constants |
| `tools/topic_classifier.py` | Topic → category classifier |
| `manual_cookies.py` | Google cookie import for Flow auth |
| `gemini-omni-prompt-generator.skill` | Omni prompt format rules |
| `auth.json` | Playwright browser state (gitignored) |
| `.env` | API keys (gitignored) |

## Common Operations

### Process Next Topic
```
1. Read AGENTS.md
2. Read first topic from topics.txt
3. Run Phase 1-4 in order
4. Mark topic as done
```

### Fix Expired Cookies
```
Ask user to run: python manual_cookies.py
Then paste fresh cookies from browser
```

### Force Category Override
Add prefix to topic in topics.txt:
```
[COSMOS] Why atoms never touch
[BIOLOGY] Why your fingers wrinkle in water
```

## Architecture Rationale

- **No CrewAI**: Direct Antigravity orchestration is more reliable than CrewAI agent chains
- **Human review gate**: Prevents wasting expensive Flow credits on bad scripts
- **Per-clip duration switching**: Different clips need different timing (hook = 4s, mechanism = 8s)
- **Extend mode**: Maintains visual continuity between clips for seamless stitching
- **Dual captions**: YouTube needs short titles, Instagram needs rich descriptions
