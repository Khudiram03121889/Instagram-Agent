import json
import unittest

from main import validate_script_clips, validate_story_blueprint
from tools.browser_tool import (
    VideoGenerationTool,
    _ensure_flow_editor_ready,
    _submission_signals,
    _verify_flow_settings,
)


def make_friction_blueprint():
    return {
        "topic_angle": "Why rubbing your hands makes them warm",
        "clip_count": 5,
        "clips": [
            {
                "clip_number": 1,
                "clip_role": "Hook",
                "core_idea": "Rubbing your palms creates friction on a cold day.",
                "bridge_from_previous": "",
                "next_clip_seed": "friction",
                "viewer_takeaway": "The warmth begins with rubbing and friction.",
                "visual_anchor_terms": ["rubbing", "palms", "friction"],
            },
            {
                "clip_number": 2,
                "clip_role": "Question",
                "core_idea": "Friction raises the question of why the hands start warming.",
                "bridge_from_previous": "friction",
                "next_clip_seed": "tiny collisions",
                "viewer_takeaway": "Friction is the reason the warmth starts.",
                "visual_anchor_terms": ["friction", "palms", "tiny collisions"],
            },
            {
                "clip_number": 3,
                "clip_role": "Mechanism",
                "core_idea": "Tiny collisions push surface particles faster and create heat.",
                "bridge_from_previous": "tiny collisions",
                "next_clip_seed": "faster particles",
                "viewer_takeaway": "Heat builds when particles move faster.",
                "visual_anchor_terms": ["tiny collisions", "particles", "heat"],
            },
            {
                "clip_number": 4,
                "clip_role": "Contrast/Payoff",
                "core_idea": "Without faster particles your hands stay cold, but with them heat appears.",
                "bridge_from_previous": "faster particles",
                "next_clip_seed": "warm hands",
                "viewer_takeaway": "The contrast is cold hands versus warm hands.",
                "visual_anchor_terms": ["cold hands", "faster particles", "warm hands"],
            },
            {
                "clip_number": 5,
                "clip_role": "Personal Takeaway",
                "core_idea": "Warm hands mean your skin is feeling energy spread.",
                "bridge_from_previous": "warm hands",
                "next_clip_seed": "",
                "viewer_takeaway": "When your hands warm up, your skin is feeling energy spread.",
                "visual_anchor_terms": ["warm hands", "skin", "energy"],
            },
        ],
    }


def make_friction_script():
    return [
        {
            "clip": 1,
            "voice_text": "You rub your palms on a cold morning. That rubbing creates friction between your hands.",
        },
        {
            "clip": 2,
            "voice_text": "Because of that friction, tiny collisions start building. So why do those tiny collisions feel warm?",
        },
        {
            "clip": 3,
            "voice_text": "Those tiny collisions push surface particles faster. Faster particles turn rubbing into rising heat.",
        },
        {
            "clip": 4,
            "voice_text": "Without those faster particles, your hands stay cold. With them, warm hands appear almost immediately.",
        },
        {
            "clip": 5,
            "voice_text": "That means your warm hands are not magic. Your skin is feeling energy spread outward.",
        },
    ]


def make_six_clip_blueprint():
    return {
        "topic_angle": "Why a fresh name slips from memory",
        "clip_count": 6,
        "clips": [
            {
                "clip_number": 1,
                "clip_role": "Hook",
                "core_idea": "You forget a name right after hearing it.",
                "bridge_from_previous": "",
                "next_clip_seed": "fresh name",
                "viewer_takeaway": "The problem starts the moment the fresh name arrives.",
                "visual_anchor_terms": ["fresh name", "face", "sound"],
            },
            {
                "clip_number": 2,
                "clip_role": "Question",
                "core_idea": "A fresh name can disappear before it settles.",
                "bridge_from_previous": "fresh name",
                "next_clip_seed": "brief holding",
                "viewer_takeaway": "The missing step happens before memory locks in.",
                "visual_anchor_terms": ["fresh name", "brief holding", "memory"],
            },
            {
                "clip_number": 3,
                "clip_role": "Mechanism Part 1",
                "core_idea": "The brain first holds the new sound only briefly.",
                "bridge_from_previous": "brief holding",
                "next_clip_seed": "attention",
                "viewer_takeaway": "The first hold is short and fragile.",
                "visual_anchor_terms": ["brief holding", "sound", "attention"],
            },
            {
                "clip_number": 4,
                "clip_role": "Mechanism Part 2",
                "core_idea": "Attention has to lock that sound into memory.",
                "bridge_from_previous": "attention",
                "next_clip_seed": "without attention",
                "viewer_takeaway": "Attention is what helps the name stay.",
                "visual_anchor_terms": ["attention", "sound", "memory"],
            },
            {
                "clip_number": 5,
                "clip_role": "Contrast/Payoff",
                "core_idea": "Without attention the name fades, but with attention it sticks.",
                "bridge_from_previous": "without attention",
                "next_clip_seed": "remembering better",
                "viewer_takeaway": "The contrast is fading versus sticking.",
                "visual_anchor_terms": ["attention", "fades", "sticks"],
            },
            {
                "clip_number": 6,
                "clip_role": "Personal Takeaway",
                "core_idea": "Remembering better starts with slowing down for one second.",
                "bridge_from_previous": "remembering better",
                "next_clip_seed": "",
                "viewer_takeaway": "You remember names better when you give attention one extra second.",
                "visual_anchor_terms": ["remembering better", "attention", "one second"],
            },
        ],
    }


def make_prompt_items():
    blueprint = make_friction_blueprint()
    script = make_friction_script()
    voice = {
        "gender": "male",
        "tone": "warm, conversational",
        "speed": 0.93,
        "pitch": "neutral-low",
        "style": "Friendly science explainer",
    }
    audio = {
        "generate_with_video": True,
        "type": "soft educational ambient",
        "volume": 0.08,
        "sfx_layers": "soft room tone, hand friction, gentle warmth",
    }
    items = []
    for index, clip in enumerate(script, start=1):
        blueprint_clip = blueprint["clips"][index - 1]
        items.append(
            {
                "clip_label": f"CLIP {index}",
                "clip_role": blueprint_clip["clip_role"],
                "voice_text": clip["voice_text"],
                "sync_terms": blueprint_clip["visual_anchor_terms"][:2],
                "visual_goal": blueprint_clip["viewer_takeaway"],
                "voice": voice,
                "background_audio": audio,
                "visual": (
                    f"Show {blueprint_clip['visual_anchor_terms'][0]} and "
                    f"{blueprint_clip['visual_anchor_terms'][1]} clearly while the viewer sees "
                    f"{blueprint_clip['core_idea']}"
                ),
                "video_style": "clear educational 3D demonstration",
                "orientation": "portrait",
                "aspect_ratio": "9:16",
            }
        )
    return items


class FakeKeyboard:
    def press(self, _key):
        return None


class FakeLocator:
    def __init__(
        self,
        text="",
        visible=True,
        count=1,
        attrs=None,
        click_callback=None,
        enabled=True,
    ):
        self.text = text
        self.visible = visible
        self._count = count
        self.attrs = attrs or {}
        self.click_callback = click_callback
        self.enabled = enabled

    @property
    def first(self):
        return self

    @property
    def last(self):
        return self

    def count(self):
        return self._count

    def is_visible(self):
        return self.visible

    def is_enabled(self):
        return self.enabled

    def get_attribute(self, name):
        return self.attrs.get(name)

    def click(self):
        if self.click_callback:
            self.click_callback(self)

    def evaluate(self, _script):
        return self.text


class FakePage:
    def __init__(self, locators):
        self.locators = locators
        self.keyboard = FakeKeyboard()

    def locator(self, selector):
        return self.locators.get(selector, FakeLocator(visible=False, count=0))

    def wait_for_timeout(self, _timeout):
        return None


class PipelineContractTests(unittest.TestCase):
    def test_blueprint_validation_accepts_six_clip_fallback(self):
        self.assertEqual(validate_story_blueprint(make_six_clip_blueprint(), True), [])

    def test_script_validation_accepts_clear_connected_script(self):
        self.assertEqual(
            validate_script_clips(make_friction_script(), make_friction_blueprint()),
            [],
        )

    def test_script_validation_rejects_vague_bridge(self):
        clips = make_friction_script()
        clips[1]["voice_text"] = "This happens for many reasons. The answer feels strange at first."
        errors = validate_script_clips(clips, make_friction_blueprint())
        self.assertTrue(any("previous clip" in error or "next clip concept" in error for error in errors))

    def test_script_validation_rejects_poetic_wording(self):
        clips = make_friction_script()
        clips[2]["voice_text"] = "Those ethereal collisions create resonance in your skin. The feeling drifts like a whisper."
        errors = validate_script_clips(clips, make_friction_blueprint())
        self.assertTrue(any("poetic" in error or "banned vague phrase" in error for error in errors))

    def test_script_validation_rejects_unclear_takeaway(self):
        clips = make_friction_script()
        clips[4]["voice_text"] = "And those warm hands still hide deeper stories. There is always more to come."
        errors = validate_script_clips(clips, make_friction_blueprint())
        self.assertTrue(any("takeaway" in error or "teaser" in error for error in errors))

    def test_script_validation_rejects_boring_filler(self):
        clips = make_friction_script()
        clips[1]["voice_text"] = (
            "Because of that friction, your palms keep rubbing. "
            "This matters more than people think."
        )
        errors = validate_script_clips(clips, make_friction_blueprint())
        self.assertTrue(any("boring" in error for error in errors))

    def test_browser_preflight_accepts_valid_sync_contract(self):
        tool = VideoGenerationTool()
        result = tool._run(
            url="https://labs.google/fx/tools/flow",
            json_content=json.dumps(make_prompt_items()),
            project_name="contract-pass",
            dry_run=True,
        )
        self.assertIn("DRY RUN PASSED", result)

    def test_browser_preflight_rejects_missing_sync_terms(self):
        items = make_prompt_items()
        del items[0]["sync_terms"]
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="missing-sync-terms",
                dry_run=True,
            )
        self.assertIn("PRE-FLIGHT VALIDATION FAILED", str(ctx.exception))

    def test_browser_preflight_rejects_invented_visual_concept(self):
        items = make_prompt_items()
        items[0]["visual"] = "Show distant galaxies and a black hole swallowing starlight."
        items[0]["visual_goal"] = "The viewer should feel cosmic wonder."
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="invented-visuals",
                dry_run=True,
            )
        self.assertIn("PRE-FLIGHT VALIDATION FAILED", str(ctx.exception))

    def test_browser_preflight_rejects_inconsistent_voice(self):
        items = make_prompt_items()
        items[1]["voice"] = {**items[1]["voice"], "tone": "formal"}
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="voice-mismatch",
                dry_run=True,
            )
        self.assertIn("voice object differs", str(ctx.exception))

    def test_browser_preflight_rejects_wrong_clip_order(self):
        items = make_prompt_items()
        items[2]["clip_role"] = "Personal Takeaway"
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="wrong-role-order",
                dry_run=True,
            )
        self.assertIn("PRE-FLIGHT VALIDATION FAILED", str(ctx.exception))

    def test_browser_preflight_rejects_wrong_clip_count(self):
        items = make_prompt_items()[:4]
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="wrong-clip-count",
                dry_run=True,
            )
        self.assertIn("PRE-FLIGHT VALIDATION FAILED", str(ctx.exception))

    def test_ensure_flow_editor_ready_uses_existing_editor(self):
        prompt_locator = FakeLocator(text="prompt", visible=True)
        page = FakePage(
            {
                "div[contenteditable='true'][data-slate-editor='true']": prompt_locator,
            }
        )
        prompt_box, selector, source = _ensure_flow_editor_ready(page)
        self.assertIs(prompt_box, prompt_locator)
        self.assertEqual(
            selector,
            "div[contenteditable='true'][data-slate-editor='true']",
        )
        self.assertEqual(source, "existing_editor")

    def test_ensure_flow_editor_ready_opens_new_project(self):
        prompt_locator = FakeLocator(text="prompt", visible=False, count=1)

        def reveal_prompt(_locator):
            prompt_locator.visible = True

        page = FakePage(
            {
                "div[contenteditable='true'][data-slate-editor='true']": prompt_locator,
                "button.fXsrxE, button.sc-a38764c7-0, button:has-text('New project')": FakeLocator(
                    visible=True,
                    click_callback=reveal_prompt,
                ),
            }
        )
        prompt_box, selector, source = _ensure_flow_editor_ready(page)
        self.assertIs(prompt_box, prompt_locator)
        self.assertEqual(
            selector,
            "div[contenteditable='true'][data-slate-editor='true']",
        )
        self.assertTrue(source.startswith("new_project:"))

    def test_verify_flow_settings_succeeds_when_controls_confirm(self):
        def activate(locator):
            locator.attrs["aria-selected"] = "true"
            locator.attrs["data-state"] = "active"

        model_control = FakeLocator(text="Veo 3.1 - Fast", attrs={"data-state": "closed"})
        locators = {
            "button[aria-haspopup='menu']": FakeLocator(
                attrs={"data-state": "open"},
            ),
            "button[id*='radix']": FakeLocator(visible=False, count=0),
            "button[role='tab']:has-text('Video'), button[id$='-trigger-VIDEO']": FakeLocator(
                attrs={"aria-selected": "true", "data-state": "active"},
                click_callback=activate,
            ),
            "button[role='tab'][id*='-trigger-PORTRAIT'], button[role='tab']:has-text('Portrait')": FakeLocator(
                attrs={"aria-selected": "false", "data-state": "inactive"},
                click_callback=activate,
            ),
            "button[role='tab']:has-text('x2')": FakeLocator(
                attrs={"aria-selected": "false", "data-state": "inactive"},
                click_callback=activate,
            ),
            "button[role='combobox']:has-text('Veo')": model_control,
            "select:has-text('Veo')": FakeLocator(visible=False, count=0),
            "[role='option']:has-text('Veo 3.1 - Fast')": FakeLocator(
                click_callback=lambda _locator: setattr(model_control, "text", "Veo 3.1 - Fast")
            ),
            "option:has-text('Veo 3.1 - Fast')": FakeLocator(visible=False, count=0),
        }
        _verify_flow_settings(FakePage(locators))

    def test_verify_flow_settings_fails_when_x2_cannot_be_verified(self):
        def activate(locator):
            locator.attrs["aria-selected"] = "true"
            locator.attrs["data-state"] = "active"

        model_control = FakeLocator(text="Veo 3.1 - Fast", attrs={"data-state": "closed"})
        locators = {
            "button[aria-haspopup='menu']": FakeLocator(attrs={"data-state": "open"}),
            "button[id*='radix']": FakeLocator(visible=False, count=0),
            "button[role='tab']:has-text('Video'), button[id$='-trigger-VIDEO']": FakeLocator(
                attrs={"aria-selected": "true", "data-state": "active"},
                click_callback=activate,
            ),
            "button[role='tab'][id*='-trigger-PORTRAIT'], button[role='tab']:has-text('Portrait')": FakeLocator(
                attrs={"aria-selected": "true", "data-state": "active"},
                click_callback=activate,
            ),
            "button[role='tab']:has-text('x2')": FakeLocator(
                attrs={"aria-selected": "false", "data-state": "inactive"},
                click_callback=lambda _locator: None,
            ),
            "button[role='combobox']:has-text('Veo')": model_control,
            "select:has-text('Veo')": FakeLocator(visible=False, count=0),
            "[role='option']:has-text('Veo 3.1 - Fast')": FakeLocator(
                click_callback=lambda _locator: setattr(model_control, "text", "Veo 3.1 - Fast")
            ),
            "option:has-text('Veo 3.1 - Fast')": FakeLocator(visible=False, count=0),
        }
        with self.assertRaises(RuntimeError) as ctx:
            _verify_flow_settings(FakePage(locators))
        self.assertIn("x2 duration", str(ctx.exception))

    def test_submission_signals_detect_click_submission(self):
        before = {
            "prompt_text": "example prompt",
            "button_enabled": True,
            "activity_visible": False,
            "activity_keywords": False,
            "result_count": 0,
        }
        after = {
            "prompt_text": "",
            "button_enabled": False,
            "activity_visible": True,
            "activity_keywords": False,
            "result_count": 0,
        }
        signals = _submission_signals(before, after, "example prompt")
        self.assertIn("prompt cleared", signals)
        self.assertIn("generate button disabled", signals)
        self.assertIn("generation activity detected", signals)

    def test_submission_signals_detect_keyboard_submit_path(self):
        before = {
            "prompt_text": "example prompt",
            "button_enabled": True,
            "activity_visible": False,
            "activity_keywords": False,
            "result_count": 0,
        }
        after = {
            "prompt_text": "example prompt submitted",
            "button_enabled": True,
            "activity_visible": True,
            "activity_keywords": True,
            "result_count": 1,
        }
        signals = _submission_signals(before, after, "example prompt")
        self.assertIn("prompt changed", signals)
        self.assertIn("generation activity detected", signals)
        self.assertIn("new result tile detected", signals)


if __name__ == "__main__":
    unittest.main()
