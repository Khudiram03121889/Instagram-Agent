# Instagram Science Reel Generator — Antigravity Pipeline

An AI-powered pipeline that generates short-form science education video content for Instagram Reels using Google Flow (Omni Flash).

## Architecture

```
topics.txt → Topic Classifier → Blueprint → Script → [HUMAN REVIEW] → Prompts → Google Flow → Captions → Archive
```

## Pipeline Phases

### Phase 1: Blueprint + Script (Automated → Human Gate)
1. Read next topic from `topics.txt`
2. Classify topic category (COSMOS/MIND/PHYSICS/BIOLOGY/CHEMISTRY/EARTH)
3. Generate story blueprint (5-6 clips with roles, durations, visual anchors)
4. Generate script following the blueprint
5. **STOP: Present to user for review** (clip count, timing per clip, total duration, script text)

### Phase 2: Prompt Generation (After Approval)
1. Generate cinematic Omni prompts using `gemini-omni-prompt-generator.skill` format
2. Pre-flight validation (sync terms, voice continuity, visual grounding)

### Phase 3: Browser Automation (Google Flow)
1. Load cookies from `auth.json` (via `manual_cookies.py`)
2. Open Google Flow via `/browser` agent
3. Set: **Video + Ingredients** / **9:16** / **x2** / **Omni Flash**
4. For each clip: switch duration tab → enter prompt → generate → wait
5. Clips 2+: Use Extend mode for visual continuity

### Phase 4: Post-Production
1. Generate YouTube title (≤100 chars)
2. Generate Instagram caption (rich, unlimited)
3. Archive all outputs
4. Mark topic as completed

## File Structure

```
├── AGENTS.md                  # Antigravity instructions (read every run)
├── INSTRUCTIONS.md            # Detailed pipeline rules & validation
├── topics.txt                 # Queue of pending topics
├── completed_topics.txt       # Archive of completed topics
├── pipeline_contract.py       # Validation constants (word limits, timing)
├── manual_cookies.py          # Cookie import for Google Flow auth
├── gemini-omni-prompt-generator.skill  # Omni prompt formatting rules
├── tools/
│   └── topic_classifier.py    # Science category classifier
├── outputs/
│   └── <topic-name>/          # Per-topic output folder
│       ├── story_blueprint.json
│       ├── validated_script.txt
│       ├── validated_prompts.json
│       ├── captions.txt
│       └── <topic-name>.txt   # Archived complete output
├── auth.json                  # Playwright storage state (gitignored)
├── .env                       # API keys (gitignored)
└── .gitignore
```

## Quick Start

1. Add topics to `topics.txt` (one per line)
2. Run `python manual_cookies.py` to import Google cookies
3. Let Antigravity run the pipeline (it follows `AGENTS.md`)

## Topic Classification

Topics are auto-classified into 6 cinematic categories:

| Category | Visual Style | Audio |
|----------|-------------|-------|
| COSMOS | Realistic 3D astronomical | Deep space resonance drone |
| MIND | Anatomical neural modeling | Warm neural ambient |
| PHYSICS | Physical simulation | Minimal resonant tones |
| BIOLOGY | Microscopic/3D educational | Biophilic ambient |
| CHEMISTRY | Molecular modeling | Crystalline resonance |
| EARTH | Documentary-style | Deep earth ambient |

Override with prefix: `[COSMOS] Why atoms never touch`

## Google Flow Settings (Exact)

| Setting | Value |
|---------|-------|
| Mode | Video + Ingredients |
| Orientation | 9:16 (Portrait) |
| Variations | x2 |
| Model | Omni Flash |
| Duration | Per-clip (4s/6s/8s/10s) |
