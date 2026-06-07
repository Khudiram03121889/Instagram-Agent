# INSTRUCTIONS.md — Detailed Pipeline Rules & Examples

This document provides extended rules, banned words, validation examples, and formatting templates that supplement `AGENTS.md`.

---

## Blueprint Design Rules

### Clip Roles (Strict Order)

**5-clip mode** (default):
| Clip | Role | Purpose | Typical Duration |
|------|------|---------|-----------------|
| 1 | Hook | Grab attention in first 2 seconds | 4s |
| 2 | Question | Plant the "why" curiosity | 6s |
| 3 | Mechanism | Explain how it works | 8s |
| 4 | Contrast/Payoff | Reveal or surprise | 6s |
| 5 | Personal Takeaway | Direct viewer benefit | 4s |

**6-clip mode** (complex topics only):
| Clip | Role | Purpose | Typical Duration |
|------|------|---------|-----------------|
| 1 | Hook | Grab attention | 4s |
| 2 | Question | Plant curiosity | 6s |
| 3 | Mechanism Part 1 | First half of explanation | 8s |
| 4 | Mechanism Part 2 | Second half of explanation | 8s |
| 5 | Contrast/Payoff | Reveal or surprise | 6s |
| 6 | Personal Takeaway | Direct viewer benefit | 4s |

### Duration Selection Guide
- **4s** — Punchy. Use for hooks and takeaways. Maximum impact in minimum time.
- **6s** — Standard. Use for questions and reveals. Enough room without dragging.
- **8s** — Detailed. Use for mechanism clips. Room to explain without rushing.
- **10s** — Dense. Only when 8s would genuinely feel too compressed. Rare.

### Visual Anchor Terms
✅ Good: `["neuron firing", "brain cross-section", "glowing synapse", "hand touching forehead"]`
❌ Bad: `["journey of the mind", "invisible force", "abstract transformation"]`

---

## Script Writing Rules

### Word Count Examples

**4-second clip (max 10 words):**
- ✅ "Your body sneezes at over a hundred miles per hour." (10 words)
- ❌ "When you sneeze your body launches air at over a hundred miles per hour." (15 words — FAIL)

**6-second clip (max 15 words):**
- ✅ "That reflex isn't random. Your brain triggers it to protect your eyes." (12 words)
- ❌ too long

**8-second clip (max 20 words):**
- ✅ "The pressure wave from a sneeze can reach your optic nerve. Closing your eyes prevents damage to the delicate tissue." (20 words)

### Bridging Rule (Clips 2+)
Each clip must echo at least one concept from the previous clip:
- Clip 1: "Your body **sneezes** at over a hundred miles per hour."
- Clip 2: "That **sneeze** isn't random. Your brain triggers it to protect your eyes." ← bridges "sneeze"

### Banned Words & Phrases

**Banned single words** (documentary/poetic style — avoid these):
```
behold, enigma, tapestry, symphony, secrets, mysteries,
profound, remarkable, extraordinary, fascinating, incredible,
unveil, unravel, journey, voyage, quest, frontier
```

**Banned phrases** (vague filler — avoid these):
```
"have you ever wondered", "in the vast expanse", "the truth is",
"it turns out", "believe it or not", "as we know it",
"the answer may surprise you", "since the dawn of time",
"what if I told you", "here's the thing", "let's dive in",
"buckle up", "mind-blowing", "game-changer"
```

### Full Script Division Rule
The Full Script must be the exact concatenation of all clip texts.
```
Full Script:
Your body sneezes at over a hundred miles per hour. That reflex isn't random. Your brain triggers it to protect your eyes.

Clip 1: Your body sneezes at over a hundred miles per hour.
Clip 2: That reflex isn't random. Your brain triggers it to protect your eyes.
```
If you join all clips, you MUST get the Full Script exactly.

---

## Prompt JSON Format

### Complete Clip Object Example
```json
{
  "clip_label": "CLIP 1",
  "clip_role": "Hook",
  "duration_seconds": 4,
  "voice_text": "Your body sneezes at over a hundred miles per hour.",
  "sync_terms": ["sneeze", "body", "speed particles"],
  "visual_goal": "Show the explosive force of a sneeze in slow motion",
  "voice": {
    "gender": "male",
    "tone": "warm, conversational",
    "speed": 1.02,
    "pitch": "neutral-low",
    "style": "Friendly science explainer — curious 22-year-old talking to a friend"
  },
  "background_audio": {
    "generate_with_video": true,
    "type": "biophilic ambient — soft forest drone with distant rustling underscore",
    "volume": 0.09,
    "sfx_layers": "biophilic forest breath, organic cellular pulse, distant water movement"
  },
  "visual": "[Camera]: Extreme close-up, slow-motion tracking\n[Style]: High-speed photography, clinical precision\n[Lighting]: Bright key light from left, dark background\n[Location]: Black void studio with single subject\n[Action]: A person mid-sneeze, tiny droplets exploding outward in slow motion, air particles visible\n[Extra]: A warm, conversational male narrator says: \"Your body sneezes at over a hundred miles per hour.\" Speed: 1.02. Ambient sound bed: biophilic forest breath, organic cellular pulse. Portrait 9:16. No on-screen text.",
  "video_style": "realistic high-quality microscopic imaging or accurate 3D educational models",
  "orientation": "portrait",
  "aspect_ratio": "9:16"
}
```

### Visual Prompt Rules
1. **Never ask for on-screen text** — no subtitles, labels, captions, UI text, letters, typography, or written equations
2. **sync_terms must be visual** — shown as real objects/actions, never as printed words
3. **Maintain consistency** — same world, palette, lighting, camera feel across all clips
4. **No Extend language** — don't write "continue from previous frame" in the prompt
5. **High-action** — visual prompts must result in dynamic motion, fast cuts, or camera reveals

---

## Caption Templates

### YouTube Title (≤100 characters, no hashtags)

**Formats:**
1. Question hook: `"Why do we close our eyes when we sneeze? 🤧"`
2. Fact reveal: `"Your sneeze travels at 100 mph — here's why your eyes shut 🔬"`
3. Curiosity gap: `"The real reason you can't sneeze with your eyes open 👀"`

### Instagram Caption (rich, unlimited)

**Template:**
```
[Hook line matching clip 1]

[2-3 lines of reflective commentary — don't re-explain the science]

[Call-to-action: save/share]

[10-15 hashtags, 2-4 emojis]
```

**Example:**
```
A hundred miles per hour. That's how fast your sneeze travels. 🤧

Your brain shuts your eyes to protect them from the pressure wave.
It's not a choice — it's a reflex older than language itself.

Save this for someone who thinks they can sneeze with their eyes open. 👀

#science #biology #sneeze #humanBody #sciencefacts
#reflex #brainfacts #education #learnontiktok
#reels #viral #mindblown #didyouknow
```

---

## Output File Formats

### story_blueprint.json
```json
{
  "topic_angle": "Why you close your eyes when you sneeze",
  "clip_count": 5,
  "clips": [
    {
      "clip_number": 1,
      "clip_role": "Hook",
      "core_idea": "...",
      "bridge_from_previous": "N/A (first clip)",
      "next_clip_seed": "...",
      "viewer_takeaway": "...",
      "visual_anchor_terms": ["...", "..."],
      "hook_pattern": "...",
      "retention_reason": "...",
      "visual_premise": "...",
      "camera_plan": "...",
      "duration_seconds": 4,
      "viewer_emotion": "..."
    }
  ]
}
```

### validated_script.txt
```
Title: Why You Close Your Eyes When You Sneeze

Full Script:
[continuous narration]

Clip 1:
[text]

Clip 2:
[text]

...
```

### captions.txt
```
--- YOUTUBE TITLE ---
Why do we close our eyes when we sneeze? 🤧

--- INSTAGRAM CAPTION ---
A hundred miles per hour. That's how fast your sneeze travels. 🤧
...
```

---

## Error Recovery

### If blueprint validation fails
- Check clip count (must be 5 or 6)
- Check role sequence order
- Check duration_seconds values (must be 4/6/8/10)
- Check total duration (must be 24-36s)
- Regenerate with specific fix instructions

### If script validation fails
- Check word counts per clip against duration limits
- Check bridging between clips
- Check for banned words/phrases
- Fix ONLY the failing clips, preserve passing ones

### If prompt pre-flight fails
- Check voice/audio consistency across clips
- Check sync_term grounding
- Check for forbidden on-screen text requests
- Regenerate JSON for failing clips only

### If Google Flow auth fails
- Ask user to run `python manual_cookies.py`
- User must paste fresh cookies from browser
- Re-run Phase 3 after cookies are updated
