"""
topic_classifier.py
--------------------
Classifies an Instagram Reel science topic into one of 6 cinematic science
categories. Returns a "topic profile" dict that is injected into every agent
and task in the pipeline, ensuring:

  - Unique video_style per science domain
  - Topic-matched audio atmosphere
  - A LOCKED voice profile (identical for every clip in the same project)
    so that the stitched video sounds like one continuous piece with the same narrator.

Categories
----------
  COSMOS   — space, atoms, energy, matter, universe
  MIND     — brain, perception, memory, consciousness, senses
  PHYSICS  — time, motion, relativity, quantum mechanics, laws
  BIOLOGY  — cells, DNA, evolution, life, organisms
  CHEMISTRY — molecules, reactions, bonds, states of matter
  EARTH    — climate, geology, oceans, ecosystems, weather
"""

import re

# ---------------------------------------------------------------------------
# VISUAL DNA REGISTRY
# Each category defines the complete aesthetic "genome" for its topics.
# ---------------------------------------------------------------------------
VISUAL_DNA = {
    "COSMOS": {
        "category": "COSMOS",
        "cinematic_reference": "Interstellar, Cosmos: A Spacetime Odyssey",
        "video_style": (
            "deep-space cinematic, ultra-dark void background with soft "
            "bioluminescent particle fields, slow orbital drift, IMAX-grade depth"
        ),
        "color_palette": (
            "deep indigo and midnight black with cold blue highlights, "
            "occasional warm amber nebula glow"
        ),
        "motion_language": (
            "imperceptibly slow drift — particles and structures move as if "
            "pulled by gravity, never rushed"
        ),
        "audio_type": "deep space resonance drone — low sub-bass hum with crystalline overtones",
        "audio_volume": 0.10,
        "narrator_delivery": "Hushed, reverent",
        "sfx_layers": "deep space resonance drone, stellar particle crackle, vacuum silence",
        "keywords": [
            "universe", "space", "atom", "atoms", "energy", "matter",
            "empty", "void", "particle", "quantum", "cosmic", "galaxy",
            "star", "orbit", "gravity", "dark matter", "black hole",
            "frozen", "solid", "light", "photon", "wave"
        ],
    },
    "MIND": {
        "category": "MIND",
        "cinematic_reference": "Her (visual warmth), Inside Out (neural imagery)",
        "video_style": (
            "soft neural-glow cinematic, warm amber and rose neuron networks "
            "pulsing gently against deep darkness, macro-close-up depth-of-field"
        ),
        "color_palette": (
            "warm amber, soft rose, muted gold — connective tissue of thought "
            "rendered as gentle glowing threads"
        ),
        "motion_language": (
            "slow rhythmic pulse — neural bursts that travel like gentle ripples "
            "across still water, organic and breathing"
        ),
        "audio_type": "warm neural ambient — soft sine-wave tones with gentle rhythmic pulse",
        "audio_volume": 0.08,
        "narrator_delivery": "Curious, intimate",
        "sfx_layers": "warm neural sine tone, soft rhythmic neural pulse, gentle breath",
        "keywords": [
            "brain", "mind", "perception", "perceive", "memory", "memories",
            "conscious", "consciousness", "think", "thought", "feel", "sense",
            "see", "reality", "edit", "fill", "predict", "illusion", "neuron",
            "cortex", "experience", "aware", "attention"
        ],
    },
    "PHYSICS": {
        "category": "PHYSICS",
        "cinematic_reference": "The Theory of Everything, Tenet (abstract time visuals)",
        "video_style": (
            "abstract geometric cinematic, precise clean lines and symmetric "
            "forms on dark slate, slow Newtonian motion contrasted with "
            "quantum probability clouds — mathematical beauty made visible"
        ),
        "color_palette": (
            "cold platinum and deep slate gray with electric-blue light traces, "
            "occasional sharp white geometric forms"
        ),
        "motion_language": (
            "deliberate and precise — objects move with Newtonian clarity, "
            "then dissolve into probability at the quantum scale"
        ),
        "audio_type": "minimal resonant tones — single pure frequencies that decay slowly into silence",
        "audio_volume": 0.09,
        "narrator_delivery": "Precise, measured",
        "sfx_layers": "low resonant frequency hum, electric field crackle, minimal silence",
        "keywords": [
            "time", "motion", "speed", "relativity", "relativistic", "force",
            "mass", "inertia", "law", "laws", "thermodynamics", "entropy",
            "quantum", "wave", "particle", "momentum", "velocity", "friction",
            "spin", "field", "electromagnetism", "gravity", "loop", "cycle"
        ],
    },
    "BIOLOGY": {
        "category": "BIOLOGY",
        "cinematic_reference": "Planet Earth II (macro nature), Inner Life of the Cell",
        "video_style": (
            "microscopic life cinematic, warm earth-tone environments with "
            "organic forms — cells, membranes, branching structures — rendered "
            "in lush macro depth-of-field with natural golden-hour lighting"
        ),
        "color_palette": (
            "warm olive greens, earthy terracottas, deep moss — the palette "
            "of living things, breathing and warm"
        ),
        "motion_language": (
            "organic and growth-like — slow unfurling, pulsing, dividing; "
            "nothing is mechanical, everything breathes"
        ),
        "audio_type": "biophilic ambient — soft forest drone with distant rustling underscore",
        "audio_volume": 0.09,
        "narrator_delivery": "Warm, wondering",
        "sfx_layers": "biophilic forest breath, organic cellular pulse, distant water movement",
        "keywords": [
            "cell", "cells", "dna", "gene", "genes", "evolution", "evolve",
            "life", "organism", "bacteria", "virus", "body", "blood", "heart",
            "nerve", "protein", "mitochondria", "atp", "photosynthesis",
            "grow", "divide", "reproduce", "species", "adapt", "survival",
            "sleep", "breath", "breathe", "immune", "skin"
        ],
    },
    "CHEMISTRY": {
        "category": "CHEMISTRY",
        "cinematic_reference": "Breaking Bad (lab aesthetics), Shots of Science series",
        "video_style": (
            "crystalline precision cinematic, studio-clean white and transparent "
            "glass environments — molecular structures float and bond in "
            "slow-motion, fluid diffusion captured at macro scale"
        ),
        "color_palette": (
            "clinical white and transparent glass with saturated accent colors "
            "wherever reactions occur — electric cyan, vivid crimson, molten gold"
        ),
        "motion_language": (
            "precise and reactive — slow diffusion builds to a crystalline snap, "
            "bonds form and break with elegant clarity"
        ),
        "audio_type": "crystalline resonance — glass harmonic tones with subtle chemical fizz undertone",
        "audio_volume": 0.10,
        "narrator_delivery": "Focused, deliberate",
        "sfx_layers": "crystalline glass harmonic ring, liquid surface tension fizz, molecular resonance",
        "keywords": [
            "molecule", "molecules", "bond", "bonds", "reaction", "react",
            "element", "compound", "chemical", "chemistry", "acid", "base",
            "electron", "proton", "neutron", "ion", "ph", "solution",
            "dissolve", "solid", "liquid", "gas", "plasma", "boil", "freeze",
            "crystal", "polymer", "carbon", "hydrogen", "oxygen", "water"
        ],
    },
    "EARTH": {
        "category": "EARTH",
        "cinematic_reference": "Planet Earth (BBC), Home (Yann Arthus-Bertrand)",
        "video_style": (
            "epic aerial nature cinematic, sweeping ultra-wide landscapes "
            "transitioning from macro geological detail to planetary overview — "
            "slow time-lapse that makes geological or climate forces visible"
        ),
        "color_palette": (
            "rich cerulean ocean blues, warm desert ochres, deep forest greens "
            "against cloud-white — the full living palette of the planet"
        ),
        "motion_language": (
            "vast and slow — erosion, weather, ocean currents, cloud formation; "
            "forces that work across centuries compressed into seconds"
        ),
        "audio_type": "deep earth ambient — wind, resonant geological hum, distant ocean beneath silence",
        "audio_volume": 0.11,
        "narrator_delivery": "Epic, grounded",
        "sfx_layers": "deep geological rumble, wind through open landscape, distant ocean movement",
        "keywords": [
            "earth", "climate", "geology", "ocean", "sea", "ecosystem", "biome",
            "weather", "atmosphere", "carbon", "water", "soil", "erosion",
            "volcano", "plate", "tectonic", "ice", "glacier", "forest",
            "biodiversity", "species", "habitat", "coral", "reef", "rain",
            "drought", "flood", "temperature", "global"
        ],
    },
}

# ============================================================
# LOCKED VOICE PROFILE
# One consistent narrator per project. This is set ONCE here
# and NEVER changes between clips within the same run.
# This ensures the stitched video sounds like one narrator
# telling one continuous story.
# ============================================================
LOCKED_VOICE_PROFILE = {
    "gender": "male",
    "tone": "calm, documentary",
    "speed": 0.93,
    "pitch": "neutral-low",
    "style": "David Attenborough measured calm — thoughtful, never rushed, never excited"
}


def classify_topic(topic: str) -> dict:
    """
    Classifies a science topic string into a cinematic category
    and returns the full topic profile with Visual DNA and locked voice.

    Parameters
    ----------
    topic : str
        The science topic, e.g. "Why atoms never touch"

    Returns
    -------
    dict
        A topic profile containing:
        - category (str)
        - cinematic_reference (str)
        - video_style (str)
        - color_palette (str)
        - motion_language (str)
        - audio_type (str)
        - audio_volume (float)
        - voice_profile (dict)  ← LOCKED, same for all clips
        - topic (str)             ← original topic echoed back
    """
    topic_lower = topic.lower()
    
    # Score each category by counting keyword matches
    scores = {}
    for category_name, dna in VISUAL_DNA.items():
        score = 0
        for keyword in dna["keywords"]:
            # Use word-boundary matching to avoid partial hits
            if re.search(r'\b' + re.escape(keyword) + r'\b', topic_lower):
                score += 1
        scores[category_name] = score

    # Pick the highest-scoring category
    best_category = max(scores, key=scores.get)
    
    # If no keyword matched at all, fall back to COSMOS (safe default for abstract science)
    if scores[best_category] == 0:
        best_category = "COSMOS"

    dna = VISUAL_DNA[best_category]

    profile = {
        "topic": topic,
        "category": dna["category"],
        "cinematic_reference": dna["cinematic_reference"],
        "video_style": dna["video_style"],
        "color_palette": dna["color_palette"],
        "motion_language": dna["motion_language"],
        "audio_type": dna["audio_type"],
        "audio_volume": dna["audio_volume"],
        # CRITICAL: voice is locked here once per topic run.
        # Every clip in this project will use the exact same
        # voice settings. Do NOT change this between clips.
        "voice_profile": LOCKED_VOICE_PROFILE.copy(),
        "narrator_delivery": dna["narrator_delivery"],
        "sfx_layers": dna["sfx_layers"],
    }

    return profile


def format_profile_for_agent(profile: dict) -> str:
    """
    Formats the topic profile into a human-readable string
    for injection into CrewAI task descriptions.
    """
    vp = profile["voice_profile"]
    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOPIC PROFILE — MANDATORY: ALL AGENTS MUST USE THIS EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Topic    : {profile['topic']}
Category : {profile['category']}
Reference: {profile['cinematic_reference']}

VISUAL DNA (use for ALL clips — do not invent your own style):
  video_style    : {profile['video_style']}
  color_palette  : {profile['color_palette']}
  motion_language: {profile['motion_language']}

AUDIO DNA (use for ALL clips — same audio = seamless stitched video):
  audio_type       : {profile['audio_type']}
  audio_volume     : {profile['audio_volume']}
  sfx_layers       : {profile['sfx_layers']}
  narrator_delivery: {profile['narrator_delivery']}

VOICE PROFILE — LOCKED FOR THIS PROJECT (DO NOT CHANGE BETWEEN CLIPS):
  gender : {vp['gender']}
  tone   : {vp['tone']}
  speed  : {vp['speed']}
  pitch  : {vp['pitch']}
  style  : {vp['style']}

CRITICAL VOICE RULE: Every clip in this project MUST use the EXACT SAME
voice settings. This is what makes the stitched video sound like one
continuous narrator telling one story — not separate disconnected clips.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()
