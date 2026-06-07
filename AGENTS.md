# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is an **Instagram Science Reel Generator** that uses CrewAI agents to create short-form educational video content. The pipeline generates scripts, video prompts, and captions for 5-6 clip reels (40-50 seconds) that explain scientific concepts clearly and engagingly.

**Architecture**: Multi-stage AI agent pipeline with local validation gates to prevent expensive video generation errors.

## Core Pipeline (Sequential Stages)

The system runs through these stages in strict order:

1. **Story Blueprint Designer** → Plans the reel structure (5 or 6 clips with defined roles)
2. **Script Writer** → Writes narration following the blueprint
3. **Prompt Engineer** → Converts script into JSON video generation prompts
4. **Browser Operator** → Automates Google Flow to generate videos
5. **Caption Writer** + **Archivist** → Creates Instagram caption and archives outputs

**Critical**: Each stage has local validation that blocks progression if quality rules fail. This prevents wasting Flow credits on bad scripts.

## Running the Pipeline

### Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests (ALWAYS run before making changes to validation logic)
python -m pytest tests/test_pipeline_contract.py -v

# Dry run (validates locally without launching Flow)
python main.py --flow-dry-run

# Live run (launches Google Flow after local validation passes)
python main.py --flow-live

# Quality mode (increases local repair attempts)
python main.py --flow-live --quality
```

### Environment Configuration

Key `.env` variables:
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL_SCRIPT` (default: gpt-5.4)
- `OPENAI_MODEL_SUPPORT` (default: gpt-5.4-mini)
- `MAX_FULL_SCRIPT_REWRITES` (default: 0, controls full script regeneration attempts)
- `MAX_LOCAL_REPAIR_ATTEMPTS` (default: 2, controls targeted fix attempts)
- `FLOW_DRY_RUN` (default: false)
- `ALLOW_SIXTH_CLIP` (default: true)
- `CHROME_USER_DATA_DIR` and `CHROME_PROFILE` (for browser auth)

## Critical Architecture Patterns

### Pipeline Contract System

The codebase enforces a **strict contract** between stages to ensure quality:

1. **Blueprint Validation** (`validate_story_blueprint`):
   - Enforces role sequences: 5-clip or 6-clip modes only
   - Required fields: `clip_number`, `clip_role`, `core_idea`, `bridge_from_previous`, `next_clip_seed`, `viewer_takeaway`, `visual_anchor_terms`
   - See `pipeline_contract.py` for constants

2. **Script Validation** (`validate_script_clips`):
   - Word limits: 12-25 words per clip (STRICT, fails at 26+)
   - Timing: Each clip must fit in 8 seconds @ 2 words/second
   - Bridging: Clips 2+ must echo concepts from previous clip
   - Banned words/phrases to avoid documentary-style language
   - Full Script Division Rule: clips must exactly match Full Script when joined

3. **Prompt JSON Validation** (in `browser_tool.py`):
   - Voice continuity: ALL clips must have identical voice settings
   - Audio continuity: ALL clips must have identical background_audio settings
   - Sync term grounding: `sync_terms` must appear in both narration AND visual
   - Speed minimum: voice.speed must be >= 1.0

### Local Repair vs Full Rewrite Strategy

When validation fails:

1. **Deterministic Salvage** (`deterministic_salvage_failed_clips`):
   - Uses blueprint's `core_idea` directly for failed clips
   - Adds minimal bridging tokens from `next_clip_seed`
   - Strips to word limits mechanically

2. **LLM Repair** (if salvage fails):
   - Targeted repair for specific failed clips only
   - Uses `script_retry_guidance` with exact error messages

3. **Full Rewrite** (last resort):
   - Controlled by `MAX_FULL_SCRIPT_REWRITES`
   - Regenerates entire script from blueprint

## Key Validation Files

- `pipeline_contract.py`: Constants for word limits, timing, clip counts
- `tests/test_pipeline_contract.py`: Contract tests (run these before changing validators)
- `main.py`: All validation functions and pipeline orchestration
- `tools/browser_tool.py`: Pre-flight validation before Flow launch

## Topic Classification System

`tools/topic_classifier.py` classifies topics into 6 categories:
- COSMOS, MIND, PHYSICS, BIOLOGY, CHEMISTRY, EARTH

Each category has:
- `video_style`: Visual instructions for that science domain
- `audio_type`, `sfx_layers`: Category-specific soundscape
- `LOCKED_VOICE_PROFILE`: Single narrator voice for entire project

**Prefix override**: Topics can force category via `[COSMOS] topic text` syntax.

## Working with Validation Logic

### Before Modifying Validators

1. **Run existing tests**: `python -m pytest tests/test_pipeline_contract.py -v`
2. **Check contract constants**: `pipeline_contract.py` defines all limits
3. **Understand the failure**: Read error messages in `validate_script_clips()` or `validate_story_blueprint()`

### Adding New Validation Rules

1. Add rule to appropriate validator in `main.py`
2. Add test case to `tests/test_pipeline_contract.py`
3. Update `pipeline_contract.py` if adding new constants
4. Update `INSTRUCTIONS.md` if agents need to know the rule

### Common Validation Patterns

- **Word count**: `word_count(text)` uses regex tokenization
- **Overlap checking**: `overlap_count(text, reference)` for bridge validation
- **Banned content**: `has_phrase(text, BANNED_PHRASES)` for phrase detection
- **Clip extraction**: `extract_failed_clips(error_text)` parses "Clip N" from errors

## Agent Configuration

Agents and tasks are defined in YAML:
- `config/agents.yaml`: Role, goal, backstory for each agent
- `config/tasks.yaml`: Task descriptions with placeholder injection

**Dynamic injection**: `{topic}` and `{topic_profile}` are replaced at runtime.

## Browser Automation

`tools/browser_tool.py` controls Google Flow via Playwright:

- **Login**: Uses `auth.json` (via `manual_cookies.py`) or Chrome profile cookies
- **Settings verification**: Confirms Veo 3.1 Fast, Portrait, 2X variant
- **Extend mode**: Clips 2+ use "Extend" to continue from previous frame
- **Submission detection**: Monitors page state changes to confirm generation started

**Pre-flight checks** block Flow launch if validation fails (saves credits).

## Output Structure

```
outputs/
  <topic-name>/
    story_blueprint.json       # Approved blueprint
    validated_script.txt       # Final script after validation
    validated_prompts.json     # JSON sent to Flow
    quality_report.json        # Validation results (if dry run)
    <topic-name>.txt          # Archived script + prompts + caption
```

## Special Files

- `topics.txt`: Queue of topics to process (first line is auto-selected)
- `completed_topics.txt`: Archive of finished topics
- `INSTRUCTIONS.md`: Full agent instructions (copy of README.md.md content)
- `pipeline_contract.py`: Single source of truth for validation constants

## Common Development Tasks

### Add a new banned word/phrase

1. Add to `BANNED_SCRIPT_WORDS` or `BANNED_SCRIPT_PHRASES` in `main.py`
2. Write test in `test_pipeline_contract.py` (see `test_script_validation_rejects_poetic_wording`)
3. Run tests to confirm

### Change word limits

1. Update constants in `pipeline_contract.py` (e.g., `SCRIPT_MAX_WORDS`)
2. Update agent instructions in `config/agents.yaml` if needed
3. Run full test suite

### Debug validation failures

1. Check `outputs/<topic-name>/quality_report.json` for detailed errors
2. Read the exact clip text that failed
3. Look at the validator function raising the error
4. Check if salvage was attempted (`deterministic_salvage_failed_clips`)

### Test a single validation function

```python
from main import validate_script_clips, extract_script_title_full_and_clips

# Parse a script
title, full_script, clips = extract_script_title_full_and_clips(script_text)

# Run validation
errors = validate_script_clips(clips, blueprint, full_script)
print(errors)
```

## Testing Philosophy

- **Contract tests** ensure pipeline stages enforce quality gates
- **Mock browser tests** verify automation logic without launching browsers
- **Validation rejection tests** confirm bad scripts are blocked early
- Run tests BEFORE pushing changes to validators

## Key Insights

1. **Validation happens in layers**: Blueprint → Script → Prompt JSON → Pre-flight (browser_tool)
2. **Voice continuity is enforced**: All clips in a project share identical voice/audio settings
3. **Bridging is semantic**: Clips must echo previous concepts using `overlap_count()` checks
4. **Word limits are HARD**: 25 words = fail (no exceptions, prevents audio truncation)
5. **Salvage before retry**: Try deterministic fixes before burning LLM calls
6. **Dry run saves money**: Always test locally before launching Flow

## Architecture Trade-offs

**Why multi-stage with validation gates?**
- Video generation is expensive (Flow credits)
- Bad scripts produce unusable videos
- Local validation costs only API tokens

**Why deterministic salvage?**
- LLM repairs can introduce new errors
- Blueprint's `core_idea` is already approved
- Mechanical fixes are predictable

**Why strict word limits?**
- 8-second audio window is fixed in video generation
- Overlong narration gets truncated mid-sentence
- Better to fail early than produce broken videos
