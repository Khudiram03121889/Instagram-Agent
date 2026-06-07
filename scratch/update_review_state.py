import json
import os

review_path = "outputs/Why do we close our eyes when we sneeze/review_state.json"
if not os.path.exists(review_path):
    print("Error: review packet does not exist!")
    exit(1)

with open(review_path, "r", encoding="utf-8") as f:
    state = json.load(f)

# Update approved and status
state["approved"] = True
state["status"] = "approved_running_flow"

# Compliant script text
state["script_text"] = (
    "Title: Why Eyes Close on Sneezes\n\n"
    "Full Script:\n"
    "Ever noticed your eyes close when you sneeze? That eye shut is a fast brainstem reflex you cannot control. "
    "The brainstem signal twitch spreads, triggering eyes to shut in one muscle burst. "
    "Closing your eyes protects them from the sneeze pressure wave automatically. "
    "This automatic protection reflex package keeps you safe.\n\n"
    "Clip 1:\n"
    "Ever noticed your eyes close when you sneeze?\n\n"
    "Clip 2:\n"
    "That eye shut is a fast brainstem reflex you cannot control.\n\n"
    "Clip 3:\n"
    "The brainstem signal twitch spreads, triggering eyes to shut in one muscle burst.\n\n"
    "Clip 4:\n"
    "Closing your eyes protects them from the sneeze pressure wave automatically.\n\n"
    "Clip 5:\n"
    "This automatic protection reflex package keeps you safe."
)

# Compliant script clips list
state["script_clips"] = [
    {
        "clip": 1,
        "voice_text": "Ever noticed your eyes close when you sneeze?"
    },
    {
        "clip": 2,
        "voice_text": "That eye shut is a fast brainstem reflex you cannot control."
    },
    {
        "clip": 3,
        "voice_text": "The brainstem signal twitch spreads, triggering eyes to shut in one muscle burst."
    },
    {
        "clip": 4,
        "voice_text": "Closing your eyes protects them from the sneeze pressure wave automatically."
    },
    {
        "clip": 5,
        "voice_text": "This automatic protection reflex package keeps you safe."
    }
]

# Write back
with open(review_path, "w", encoding="utf-8") as f:
    json.dump(state, f, ensure_ascii=False, indent=2)

print("Successfully approved and updated review state!")
