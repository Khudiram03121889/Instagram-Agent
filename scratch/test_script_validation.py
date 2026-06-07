import json
from main import validate_script_clips, clip_text_as_full_script

blueprint_path = "outputs/Why do we close our eyes when we sneeze/review_state.json"
with open(blueprint_path, "r", encoding="utf-8") as f:
    state = json.load(f)

blueprint = state["blueprint"]

custom_clips = [
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

full_script = clip_text_as_full_script(custom_clips)

errors = validate_script_clips(custom_clips, blueprint, full_script)
print("Validation Errors:")
for e in errors:
    print("-", e)
if not errors:
    print("SUCCESS: The script passes all contract validation rules!")
