from __future__ import annotations

from typing import Dict, List

SCRIPT_MIN_WORDS = 8
SCRIPT_MAX_WORDS = 20
SCRIPT_MAX_SENTENCES = 3
SCRIPT_MAX_SECONDS = 8.0
SCRIPT_WORDS_PER_SECOND = 2.5
CAPTION_MAX_CHARS = 100

MIN_CLIP_COUNT = 5
MAX_CLIP_COUNT = 6

BLUEPRINT_ROLE_SEQUENCES: Dict[int, List[str]] = {
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
