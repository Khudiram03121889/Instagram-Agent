---
name: gemini-omni-prompt-generator
description: >
  Generates fully optimized prompts for Google DeepMind's Gemini Omni model — a multimodal video/image/audio AI.
  Use this skill ALWAYS when the user asks for:
  - "Give me a Gemini Omni prompt for..."
  - "Create a prompt for Gemini Omni..."
  - "Optimized prompt for Gemini Omni to generate..."
  - "Write a Gemini Omni prompt to make a video of..."
  - "Help me prompt Gemini Omni for..."
  - Any request to generate, edit, or transform video/image/audio using Gemini Omni
  - Any request involving Google Flow, Gemini video generation, or Omni-based creation
  This skill applies to text-to-video, image-to-video, video editing, style transfer, multi-input prompts, and iterative editing scenarios.
---

# Gemini Omni Prompt Generator

## What is Gemini Omni?

Gemini Omni is Google DeepMind's multimodal generative AI model capable of creating and editing video, images, audio, and text — all through natural language. Unlike simpler models, Omni understands **world knowledge, cinematic language, and complex actions**, meaning you don't need to over-explain details — just describe your intention.

Available via: **Gemini app** and **Google Flow**

---

## Prompt Structure (The 5 Core Elements)

Every great Gemini Omni prompt should include a mix of these elements:

### 1. 🎬 Shot Framing & Camera Motion
Specify how to frame and move the camera:
- **Framing**: `extreme close-up`, `close-up`, `medium shot`, `wide shot`, `establishing shot`, `aerial shot`, `POV shot`
- **Motion**: `push in`, `pull back`, `dolly zoom`, `pan left/right`, `tilt up/down`, `tracking shot`, `crane shot`, `handheld`, `static/locked off`
- **Camera type**: `film camera`, `webcam style`, `natural smartphone zoom`, `IMAX`, `anamorphic lens`
- **Continuity**: `one continuous shot`, `oner`, `seamless cut`

### 2. 🎨 Style & Aesthetic
Tell Omni the visual feel:
- **Cinematic styles**: `cinematic`, `documentary`, `music video`, `short film`
- **Art styles**: `anime`, `claymation`, `watercolour`, `oil painting`, `flat vector`, `risograph print`, `pixel art`, `3D render`, `sketch`, `minimalist`
- **Texture descriptors**: `grainy`, `film grain`, `sharp`, `soft focus`, `high contrast`, `muted tones`, `neon electric palette`, `pastel`, `monochrome`

### 3. 💡 Lighting
Describe light source and quality:
- **Sources**: `golden hour sunlight`, `neon streetlamp`, `moonlight`, `studio lighting`, `practical lights`, `off-screen fill light`, `motivated light from window`
- **Quality**: `crisp and hard`, `soft diffused`, `warm and amber`, `cool and blue`, `ethereal glow`, `volumetric rays`, `rim lighting`, `chiaroscuro`

### 4. 📍 Location & Environment
Set the scene:
- Be evocative, not exhaustive — Omni fills in details based on world knowledge
- Example: `"a foggy Victorian alleyway"`, `"alien landscape with bioluminescent flora"`, `"minimalist white studio"`, `"crowded Mumbai local train at peak hour"`

### 5. 🎭 Action & Characters
Describe what is happening:
- Who are the characters/objects?
- How are they moving and interacting?
- What is the emotional tone of the action?
- For complex actions (e.g. parkour, classical dance), just name the action — Omni understands it across frames

---

## Prompt Templates by Use Case

### Text-to-Video (Generation from scratch)
```
[Camera]: [SHOT TYPE], [CAMERA MOTION]
[Style]: [AESTHETIC], [TEXTURE/FEEL]
[Lighting]: [LIGHT SOURCE], [LIGHT QUALITY]
[Location]: [ENVIRONMENT DESCRIPTION]
[Action]: [CHARACTERS/OBJECTS + WHAT THEY DO]
[Extra]: [TEXT RENDERING / AUDIO SYNC / SPECIAL EFFECTS if needed]
```

### Video Editing / Style Transfer
```
Edit this keeping everything the same. [SPECIFIC CHANGE ONLY].
```
*Be surgical — describe only what changes, not the whole scene again.*

### Multi-Input (Image + Video + Audio)
```
[Reference the birds/elements from <video>] [do X based on <image>].
[They move to the music from <audio>] and [do Y as the scene ends].
```

### Storyboard-Based
```
Follow this story exactly in order starting top left.
Entire story in [X] seconds. [STYLE]. [CAMERA DIRECTION].
```

### Character/Object Consistency
Generate or reference a character image first (via Nano Banana), then:
```
Keep the [character/object] from <reference image> consistent throughout. [REST OF SCENE DESCRIPTION].
```

---

## Power Techniques

### World Knowledge — Don't Over-Explain
Gemini Omni knows history, science, culture, and cinematography.
- ❌ Weak: `"show two computers, one doing things step by step and one doing many things at once"`
- ✅ Strong: `"Visualize the difference between classical computing and quantum computing"`

### Iterative Editing — Change One Thing at a Time
After generating a base video, edit naturally:
- `"Change the butterfly to a bee."`
- `"Now change the bee into a small swarm of fireflies."`
- `"Change the camera angle to be over the violinist's shoulder."`

### Sync to Audio
- `"The lights of the apartments start turning on in sync with the music."`

### Animated Text
- `"Word by word, one word on screen at a time: [words]. Each word appears with a different animated style, perfect pacing to a rhythm."`

### Cinematic Camera Moves (Use exact film terms)
| Want | Use |
|------|-----|
| Zoom in dramatically | `"dolly zoom"` or `"punch in"` |
| Follow action | `"tracking shot"` |
| Slow rise overhead | `"crane shot"` |
| Raw/shaky feel | `"handheld"` |
| No movement | `"static"` / `"locked off"` |
| Full scene in one take | `"oner"` / `"one continuous shot"` |

---

## Output Format

When a user asks for a Gemini Omni prompt, output:

1. **The Ready-to-Use Prompt** — formatted, complete, copy-paste ready
2. **Brief breakdown** — one line per element explaining the choice (only if the user seems to want to learn; skip for experienced users)
3. **Iterative edit suggestions** — 2–3 follow-up prompts they can use to refine the output in Gemini Omni

---

## Example Prompts

### Short Film Scene
```
One continuous shot. A medium shot slowly pushing in. Cinematic style with film grain and warm amber lighting from a single practical desk lamp. A dimly lit 1970s detective office in New York. A tired detective in a rumpled suit stubs out a cigarette and picks up a ringing rotary phone, pausing before answering — a look of dread crossing his face.
```

### Science Explainer Video
```
Visualize the difference between regular computing (step-by-step bits) and quantum computing (superposition of qubits). Contemporary flat-media style blending minimalist vector shapes with rich organic textures. High-contrast electric color palette of neon pinks, cyans, and limes set against a deep navy background. Stipple shading and grainy gradients add a tactile risograph-like quality. Word labels animate on screen to identify each system.
```

### Instagram Reel — Indian Street Food
```
Handheld smartphone zoom style. Slow tracking shot moving through a crowded Mumbai street food market at dusk. Warm golden practical lighting from overhead bulbs and stove flames. Steam rising off a sizzling tawa as a vendor tosses pav bhaji, sparks and spice in the air. Fast-cut montage feel with a punchy, vibrant color grade.
```

### Style Transfer
```
Create a four-part stylistic progression of the input video: begin with vibrant crayon strokes on granulated paper, transition to graphite pencil sketch with cross-hatching and 12fps line-boiling effect, morph into hyper-realistic 3D translucent glass with caustic light refractions, conclude with risograph print using a limited three-color palette and halftone textures.
```

---

## Notes

- Gemini Omni **preserves your video across multiple edits** — you only need to describe what changes
- You can combine **image + video + audio + text** in one prompt using `<reference>` syntax
- For the most consistent character/object across scenes, generate a reference image first using **Nano Banana**, then pass it to Omni
- Available at: [gemini.google.com](https://gemini.google.com) and [flow.google.com](https://flow.google.com)
- Requires a Google AI subscription (features vary by tier and geography)
