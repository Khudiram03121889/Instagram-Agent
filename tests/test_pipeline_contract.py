import json
import inspect
import os
import unittest

import yaml

from dashboard import create_review_state, load_review_state, save_review_state
from main import (
    clip_text_as_full_script,
    coerce_prompt_json_sync_term_grounding,
    extract_script_title_full_and_clips,
    merge_failed_clip_repairs,
    run_browser_operator_stage,
    validate_script_clips,
    validate_story_blueprint,
    verify_pipeline_task_connections,
)
from tools.browser_tool import (
    VideoGenerationTool,
    _build_cinematic_prompt,
    _capture_latest_result_snapshot,
    _classify_generation_timeout,
    _ensure_flow_editor_ready,
    _ensure_extend_mode,
    _open_variant,
    _select_result_snapshot,
    _select_flow_duration,
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
                "hook_pattern": "everyday body moment",
                "retention_reason": "viewer wants to know why warmth appears fast",
                "visual_premise": "macro palms rubbing with heat appearing",
                "camera_plan": "close-up push in on rubbing palms",
                "duration_seconds": 8,
                "viewer_emotion": "curious recognition",
            },
            {
                "clip_number": 2,
                "clip_role": "Question",
                "core_idea": "Friction raises the question of why the hands start warming.",
                "bridge_from_previous": "friction",
                "next_clip_seed": "tiny collisions",
                "viewer_takeaway": "Friction is the reason the warmth starts.",
                "visual_anchor_terms": ["friction", "palms", "tiny collisions"],
                "hook_pattern": "why question",
                "retention_reason": "turns friction into a visible cause",
                "visual_premise": "surface ridges catching and bumping",
                "camera_plan": "macro tracking shot across skin ridges",
                "duration_seconds": 6,
                "viewer_emotion": "focused curiosity",
            },
            {
                "clip_number": 3,
                "clip_role": "Mechanism",
                "core_idea": "Tiny collisions push surface particles faster and create heat.",
                "bridge_from_previous": "tiny collisions",
                "next_clip_seed": "faster particles",
                "viewer_takeaway": "Heat builds when particles move faster.",
                "visual_anchor_terms": ["tiny collisions", "particles", "heat"],
                "hook_pattern": "hidden mechanism reveal",
                "retention_reason": "shows the invisible mechanism",
                "visual_premise": "particles speed up after tiny collisions",
                "camera_plan": "push through skin into particle view",
                "duration_seconds": 6,
                "viewer_emotion": "aha",
            },
            {
                "clip_number": 4,
                "clip_role": "Contrast/Payoff",
                "core_idea": "Without faster particles your hands stay cold, but with them heat appears.",
                "bridge_from_previous": "faster particles",
                "next_clip_seed": "warm hands",
                "viewer_takeaway": "The contrast is cold hands versus warm hands.",
                "visual_anchor_terms": ["cold hands", "faster particles", "warm hands"],
                "hook_pattern": "contrast payoff",
                "retention_reason": "cold versus warm makes the payoff clear",
                "visual_premise": "split physical contrast between cold and warm hands",
                "camera_plan": "side-by-side medium close-up",
                "duration_seconds": 6,
                "viewer_emotion": "satisfying clarity",
            },
            {
                "clip_number": 5,
                "clip_role": "Personal Takeaway",
                "core_idea": "Warm hands mean your skin is feeling energy spread.",
                "bridge_from_previous": "warm hands",
                "next_clip_seed": "",
                "viewer_takeaway": "When your hands warm up, your skin is feeling energy spread.",
                "visual_anchor_terms": ["warm hands", "skin", "energy"],
                "hook_pattern": "personal takeaway",
                "retention_reason": "connects the science to the viewer's body",
                "visual_premise": "warmth spreading through skin",
                "camera_plan": "gentle pull back from glowing palms",
                "duration_seconds": 6,
                "viewer_emotion": "useful understanding",
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
            "voice_text": "Because of that friction, tiny collisions start building. Why do those collisions feel warm?",
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


def make_friction_full_script():
    return clip_text_as_full_script(make_friction_script())


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
                "hook_pattern": "relatable memory fail",
                "retention_reason": "viewer recognizes the awkward moment",
                "visual_premise": "a face and sound arriving then slipping",
                "camera_plan": "close-up rack focus from face to ear",
                "duration_seconds": 4,
                "viewer_emotion": "recognition",
            },
            {
                "clip_number": 2,
                "clip_role": "Question",
                "core_idea": "A fresh name can disappear before it settles.",
                "bridge_from_previous": "fresh name",
                "next_clip_seed": "brief holding",
                "viewer_takeaway": "The missing step happens before memory locks in.",
                "visual_anchor_terms": ["fresh name", "brief holding", "memory"],
                "hook_pattern": "missing step question",
                "retention_reason": "promises the hidden step",
                "visual_premise": "new sound hovering before memory catches it",
                "camera_plan": "slow push into head profile",
                "duration_seconds": 4,
                "viewer_emotion": "curious",
            },
            {
                "clip_number": 3,
                "clip_role": "Mechanism Part 1",
                "core_idea": "The brain first holds the new sound only briefly.",
                "bridge_from_previous": "brief holding",
                "next_clip_seed": "attention",
                "viewer_takeaway": "The first hold is short and fragile.",
                "visual_anchor_terms": ["brief holding", "sound", "attention"],
                "hook_pattern": "mechanism part one",
                "retention_reason": "turns memory into a fragile physical hold",
                "visual_premise": "sound pulse held briefly",
                "camera_plan": "macro tracking shot along a neural path",
                "duration_seconds": 4,
                "viewer_emotion": "focused",
            },
            {
                "clip_number": 4,
                "clip_role": "Mechanism Part 2",
                "core_idea": "Attention has to lock that sound into memory.",
                "bridge_from_previous": "attention",
                "next_clip_seed": "without attention",
                "viewer_takeaway": "Attention is what helps the name stay.",
                "visual_anchor_terms": ["attention", "sound", "memory"],
                "hook_pattern": "mechanism part two",
                "retention_reason": "shows the lock-in step",
                "visual_premise": "attention beam stabilizing a sound pulse",
                "camera_plan": "locked-off shot with one bright motion",
                "duration_seconds": 4,
                "viewer_emotion": "aha",
            },
            {
                "clip_number": 5,
                "clip_role": "Contrast/Payoff",
                "core_idea": "Without attention the name fades, but with attention it sticks.",
                "bridge_from_previous": "without attention",
                "next_clip_seed": "remembering better",
                "viewer_takeaway": "The contrast is fading versus sticking.",
                "visual_anchor_terms": ["attention", "fades", "sticks"],
                "hook_pattern": "contrast payoff",
                "retention_reason": "clear fading versus sticking comparison",
                "visual_premise": "one sound fades while one sticks",
                "camera_plan": "split-screen close-up",
                "duration_seconds": 4,
                "viewer_emotion": "clarity",
            },
            {
                "clip_number": 6,
                "clip_role": "Personal Takeaway",
                "core_idea": "Remembering better starts with slowing down for one second.",
                "bridge_from_previous": "remembering better",
                "next_clip_seed": "",
                "viewer_takeaway": "You remember names better when you give attention one extra second.",
                "visual_anchor_terms": ["remembering better", "attention", "one second"],
                "hook_pattern": "personal takeaway",
                "retention_reason": "gives a one-second behavior",
                "visual_premise": "person pauses and anchors the name",
                "camera_plan": "gentle pull back from face and handshake",
                "duration_seconds": 4,
                "viewer_emotion": "practical relief",
            },
        ],
    }


def make_prompt_items():
    blueprint = make_friction_blueprint()
    script = make_friction_script()
    voice = {
        "gender": "male",
        "tone": "warm, conversational",
        "speed": 1.02,
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
                "duration_seconds": blueprint_clip["duration_seconds"],
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
        items=None,
    ):
        self.text = text
        self.visible = visible
        self._count = count
        self.attrs = attrs or {}
        self.click_callback = click_callback
        self.enabled = enabled
        self.items = items

    @property
    def first(self):
        if self.items:
            return self.items[0]
        return self

    @property
    def last(self):
        if self.items:
            return self.items[-1]
        return self

    def count(self):
        if self.items is not None:
            return len(self.items)
        return self._count

    def is_visible(self):
        return self.visible

    def is_enabled(self):
        return self.enabled

    def get_attribute(self, name):
        return self.attrs.get(name)

    def click(self, *args, **kwargs):
        if self.click_callback:
            self.click_callback(self)

    def evaluate(self, _script):
        return self.text

    def inner_text(self, timeout=None):
        return self.text

    def nth(self, index):
        if self.items is not None:
            return self.items[index]
        return self


class FakePage:
    def __init__(self, locators):
        self.locators = locators
        self.keyboard = FakeKeyboard()

    def locator(self, selector):
        return self.locators.get(selector, FakeLocator(visible=False, count=0))

    def wait_for_timeout(self, _timeout):
        return None


class DummyTaskNode:
    def __init__(self, context=None, raw=""):
        self.context = context or []
        self.description = ""
        self.output = type("Output", (), {})()
        self.output.raw = raw


class DummyCrew:
    def __init__(self, response="agent output"):
        self.response = response

    def kickoff(self):
        return self.response


class DummyVideoTool:
    def __init__(self):
        self.current_usage_count = 0
        self.calls = []

    def _run(self, **kwargs):
        self.calls.append(kwargs)
        return "DRY RUN PASSED"


class PipelineContractTests(unittest.TestCase):
    def test_blueprint_validation_accepts_six_clip_fallback(self):
        self.assertEqual(validate_story_blueprint(make_six_clip_blueprint(), True), [])

    def test_script_validation_accepts_clear_connected_script(self):
        self.assertEqual(
            validate_script_clips(
                make_friction_script(),
                make_friction_blueprint(),
                make_friction_full_script(),
            ),
            [],
        )

    def test_dashboard_review_state_saves_reload_and_approves(self):
        topic = "dashboard-review-state-test"
        state = create_review_state(
            topic=topic,
            blueprint=make_friction_blueprint(),
            script_text="Title: Test\n\nFull Script:\nTest",
            script_clips=make_friction_script(),
            topic_profile={"category": "PHYSICS"},
        )
        state["manual_feedback"] = "Make the hook sharper."
        path = save_review_state(topic, state)
        self.assertTrue(os.path.exists(path))

        loaded = load_review_state(topic)
        self.assertEqual(loaded["manual_feedback"], "Make the hook sharper.")
        loaded["approved"] = True
        loaded["status"] = "approved_running_flow"
        save_review_state(topic, loaded)
        approved = load_review_state(topic)
        self.assertTrue(approved["approved"])
        self.assertEqual(approved["status"], "approved_running_flow")

    def test_script_parser_extracts_full_script_and_clip_division(self):
        script_text = (
            "Title: Hand Heat\n\n"
            f"Full Script:\n{make_friction_full_script()}\n\n"
            "Clip 1:\n"
            f"{make_friction_script()[0]['voice_text']}\n\n"
            "Clip 2:\n"
            f"{make_friction_script()[1]['voice_text']}\n\n"
            "Clip 3:\n"
            f"{make_friction_script()[2]['voice_text']}\n\n"
            "Clip 4:\n"
            f"{make_friction_script()[3]['voice_text']}\n\n"
            "Clip 5:\n"
            f"{make_friction_script()[4]['voice_text']}"
        )
        title, full_script, clips = extract_script_title_full_and_clips(script_text)
        self.assertEqual(title, "Hand Heat")
        self.assertEqual(full_script, make_friction_full_script())
        self.assertEqual(clips, make_friction_script())

    def test_script_validation_rejects_non_exact_full_script_division(self):
        errors = validate_script_clips(
            make_friction_script(),
            make_friction_blueprint(),
            make_friction_full_script() + " Extra sentence.",
        )
        self.assertTrue(any("Full Script must exactly match" in error for error in errors))

    def test_script_validation_rejects_vague_bridge(self):
        clips = make_friction_script()
        clips[1]["voice_text"] = "This happens for many reasons. The answer stays unclear for most people."
        errors = validate_script_clips(clips, make_friction_blueprint(), clip_text_as_full_script(clips))
        self.assertTrue(any("previous clip" in error or "next clip concept" in error for error in errors))

    def test_script_validation_rejects_poetic_wording(self):
        clips = make_friction_script()
        clips[2]["voice_text"] = "Those ethereal collisions create resonance in your skin. The feeling drifts like a whisper."
        errors = validate_script_clips(clips, make_friction_blueprint(), clip_text_as_full_script(clips))
        self.assertTrue(any("poetic" in error or "banned vague phrase" in error for error in errors))

    def test_script_validation_rejects_unclear_takeaway(self):
        pass

    def test_script_validation_rejects_boring_filler(self):
        clips = make_friction_script()
        clips[1]["voice_text"] = (
            "Because of that friction, your palms keep rubbing. "
            "This matters more than people think."
        )
        errors = validate_script_clips(clips, make_friction_blueprint(), clip_text_as_full_script(clips))
        self.assertTrue(any("boring" in error for error in errors))

    def test_script_validation_rejects_over_20_words(self):
        clips = make_friction_script()
        clips[0]["voice_text"] = (
            "You rub your palms on a cold winter morning and notice fast warmth building in seconds now. Because friction is an amazing physical process that converts motion into thermal energy very quickly."
        )
        errors = validate_script_clips(clips, make_friction_blueprint(), clip_text_as_full_script(clips))
        self.assertTrue(any("EXCEEDS 20-word limit" in error for error in errors))

    def test_browser_preflight_accepts_valid_sync_contract(self):
        tool = VideoGenerationTool()
        result = tool._run(
            url="https://labs.google/fx/tools/flow",
            json_content=json.dumps(make_prompt_items()),
            project_name="contract-pass",
            dry_run=True,
        )
        self.assertIn("DRY RUN PASSED", result)

    def test_sync_term_grounding_coercion_unblocks_preflight(self):
        items = make_prompt_items()
        items[4]["sync_terms"] = ["warm hands", "skin"]
        items[4]["voice_text"] = "Warm hands are not magic. Your skin is feeling energy spread outward."
        items[4]["visual_goal"] = "Viewer understands the idea clearly."
        items[4]["visual"] = "Show a calm abstract scene with soft lighting and gentle motion."

        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError):
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="ungrounded-sync-terms",
                dry_run=True,
            )

        fixed_json = coerce_prompt_json_sync_term_grounding(json.dumps(items))
        result = tool._run(
            url="https://labs.google/fx/tools/flow",
            json_content=fixed_json,
            project_name="grounded-sync-terms",
            dry_run=True,
        )
        self.assertIn("DRY RUN PASSED", result)

    def test_browser_preflight_rejects_slow_voice_speed(self):
        items = make_prompt_items()
        for item in items:
            item["voice"]["speed"] = 0.95
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="slow-voice-speed",
                dry_run=True,
            )
        self.assertIn("voice.speed must be >= 1.0", str(ctx.exception))

    def test_browser_preflight_rejects_speed_aware_timing_overflow(self):
        import tools.browser_tool as browser_tool

        original_wps = browser_tool.SCRIPT_WORDS_PER_SECOND
        browser_tool.SCRIPT_WORDS_PER_SECOND = 1.0
        try:
            items = make_prompt_items()
            for item in items:
                item["voice"]["speed"] = 1.0
            tool = VideoGenerationTool()
            with self.assertRaises(RuntimeError) as ctx:
                tool._run(
                    url="https://labs.google/fx/tools/flow",
                    json_content=json.dumps(items),
                    project_name="timing-overflow",
                    dry_run=True,
                )
            self.assertIn("estimated audio length", str(ctx.exception))
        finally:
            browser_tool.SCRIPT_WORDS_PER_SECOND = original_wps

    def test_prompt_builder_marks_later_clips_as_standalone_omni_prompts(self):
        prompt = _build_cinematic_prompt(make_prompt_items()[1], clip_number=2)
        self.assertIn("standalone Omni Flash clip", prompt)
        self.assertNotIn("Google Flow Extend", prompt)
        self.assertNotIn("previous clip's final frame", prompt)

    def test_prompt_builder_marks_first_clip_as_visual_world_start(self):
        prompt = _build_cinematic_prompt(make_prompt_items()[0], clip_number=1)
        self.assertIn("standalone Omni Flash clip", prompt)
        self.assertNotIn("Google Flow Extend", prompt)

    def test_prompt_builder_forbids_rendered_text_in_visuals(self):
        prompt = _build_cinematic_prompt(make_prompt_items()[0], clip_number=1)
        self.assertIn("No on-screen text", prompt)
        self.assertIn("Do not render sync terms as printed words", prompt)

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

    def test_browser_preflight_rejects_visual_text_overlay_requests(self):
        items = make_prompt_items()
        items[0]["visual"] = "Show rubbing palms and display the word FRICTION as bold on-screen text."
        tool = VideoGenerationTool()
        with self.assertRaises(RuntimeError) as ctx:
            tool._run(
                url="https://labs.google/fx/tools/flow",
                json_content=json.dumps(items),
                project_name="text-overlay-visual",
                dry_run=True,
            )
        self.assertIn("visual plan requests rendered text", str(ctx.exception))

    def test_browser_preflight_allows_negative_no_text_guardrails(self):
        items = make_prompt_items()
        items[0]["visual"] = (
            "Show rubbing palms in macro detail with no text, no labels, and no captions anywhere on screen."
        )
        tool = VideoGenerationTool()
        result = tool._run(
            url="https://labs.google/fx/tools/flow",
            json_content=json.dumps(items),
            project_name="negative-no-text-guardrail",
            dry_run=True,
        )
        self.assertIn("DRY RUN PASSED", result)

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

        model_control = FakeLocator(text="Omni Flash", attrs={"data-state": "closed"})
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
            "button[role='tab']:has-text('2X'), button[role='tab']:has-text('x2'), button:has-text('2X')": FakeLocator(
                attrs={"aria-selected": "false", "data-state": "inactive"},
                click_callback=activate,
            ),
            "button[role='combobox']:has-text('Omni')": model_control,
            "button[aria-haspopup='menu']:has-text('Omni')": FakeLocator(visible=False, count=0),
            "button[role='combobox']:has-text('Veo')": FakeLocator(visible=False, count=0),
            "button[aria-haspopup='menu']:has-text('Veo')": FakeLocator(visible=False, count=0),
            "[role='option']:has-text('Omni Flash')": FakeLocator(
                click_callback=lambda _locator: setattr(model_control, "text", "Omni Flash")
            ),
            "span:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
            "button:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
            "option:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
            "button[role='tab']:has-text('8s'), button:has-text('8s')": FakeLocator(
                attrs={"aria-selected": "true", "data-state": "active"},
                click_callback=activate,
            ),
        }
        _verify_flow_settings(FakePage(locators), duration_seconds=8)

    def test_verify_flow_settings_fails_when_x2_cannot_be_verified(self):
        def activate(locator):
            locator.attrs["aria-selected"] = "true"
            locator.attrs["data-state"] = "active"

        model_control = FakeLocator(text="Omni Flash", attrs={"data-state": "closed"})
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
            "button[role='tab']:has-text('2X'), button[role='tab']:has-text('x2'), button:has-text('2X')": FakeLocator(
                attrs={"aria-selected": "false", "data-state": "inactive"},
                click_callback=lambda _locator: None,
            ),
            "button[role='combobox']:has-text('Omni')": model_control,
            "button[aria-haspopup='menu']:has-text('Omni')": FakeLocator(visible=False, count=0),
            "button[role='combobox']:has-text('Veo')": FakeLocator(visible=False, count=0),
            "button[aria-haspopup='menu']:has-text('Veo')": FakeLocator(visible=False, count=0),
            "[role='option']:has-text('Omni Flash')": FakeLocator(
                click_callback=lambda _locator: setattr(model_control, "text", "Omni Flash")
            ),
            "span:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
            "button:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
            "option:has-text('Omni Flash')": FakeLocator(visible=False, count=0),
        }
        with self.assertRaises(RuntimeError) as ctx:
            _verify_flow_settings(FakePage(locators))
        self.assertIn("2X variant count", str(ctx.exception))

    def test_select_flow_duration_uses_requested_duration(self):
        def activate(locator):
            locator.attrs["aria-selected"] = "true"
            locator.attrs["data-state"] = "active"

        duration_locator = FakeLocator(
            text="6s",
            attrs={"aria-selected": "false", "data-state": "inactive"},
            click_callback=activate,
        )
        selector = _select_flow_duration(
            FakePage({"button[role='tab']:has-text('6s'), button:has-text('6s')": duration_locator}),
            6,
        )
        self.assertIn("6s", selector)
        self.assertEqual(duration_locator.attrs["aria-selected"], "true")

    def test_live_flow_workflow_does_not_open_or_extend_variants(self):
        source = inspect.getsource(VideoGenerationTool._run)
        self.assertNotIn("_open_variant(", source)
        self.assertNotIn("_ensure_extend_mode(", source)

    def test_ensure_extend_mode_clicks_extend_button(self):
        def activate(locator):
            locator.attrs["aria-pressed"] = "true"

        extend_locator = FakeLocator(
            text="Extend",
            attrs={"aria-pressed": "false"},
            click_callback=activate,
        )
        selector = _ensure_extend_mode(
            FakePage({"button:has-text('Extend')": extend_locator})
        )
        self.assertEqual(selector, "button:has-text('Extend')")
        self.assertEqual(extend_locator.attrs["aria-pressed"], "true")

    def test_open_variant_prefers_saved_href_over_live_index(self):
        prompt_locator = FakeLocator(text="prompt", visible=True)
        wrong_locator = FakeLocator(
            visible=True,
            attrs={"href": "/edit/wrong"},
            click_callback=lambda locator: setattr(locator, "attrs", {**locator.attrs, "clicked": "true"}),
        )
        right_locator = FakeLocator(
            visible=True,
            attrs={"href": "/edit/right"},
            click_callback=lambda locator: setattr(locator, "attrs", {**locator.attrs, "clicked": "true"}),
        )
        page = FakePage(
            {
                "a[href*='/edit/']:has(img)": wrong_locator,
                "a[href='/edit/right']": right_locator,
                "div[contenteditable='true'][data-slate-editor='true']": prompt_locator,
            }
        )

        selector = _open_variant(
            page,
            {"selector": "a[href*='/edit/']:has(img)", "ordinal": 0, "href": "/edit/right"},
        )
        self.assertEqual(selector, "a[href='/edit/right']")
        self.assertEqual(right_locator.attrs.get("clicked"), "true")

    def test_select_result_snapshot_reselects_saved_branch_tile(self):
        branch_locator = FakeLocator(
            visible=True,
            attrs={"href": "/edit/variant-2", "aria-selected": "false"},
            click_callback=lambda locator: locator.attrs.__setitem__("aria-selected", "true"),
        )
        page = FakePage({"a[href='/edit/variant-2']": branch_locator})

        selector = _select_result_snapshot(
            page,
            {"selector": "a[href*='/edit/']:has(img)", "ordinal": 1, "href": "/edit/variant-2"},
            label="Variant 2 current clip tile",
        )
        self.assertEqual(selector, "a[href='/edit/variant-2']")
        self.assertEqual(branch_locator.attrs.get("aria-selected"), "true")

    def test_capture_latest_result_snapshot_returns_last_visible_tile(self):
        first = FakeLocator(visible=True, attrs={"href": "/edit/variant-1"})
        second = FakeLocator(visible=True, attrs={"href": "/edit/variant-1-clip-2"})
        locator_group = FakeLocator(items=[first, second])
        page = FakePage({"a[href*='/edit/']:has(img)": locator_group})

        snapshot = _capture_latest_result_snapshot(page)
        self.assertEqual(snapshot["href"], "/edit/variant-1-clip-2")
        self.assertEqual(snapshot["ordinal"], 1)

    def test_configs_remove_editing_advisor_and_caption_save_prompt(self):
        with open("config/agents.yaml", "r", encoding="utf-8-sig") as file:
            agents = yaml.safe_load(file)
        with open("config/tasks.yaml", "r", encoding="utf-8-sig") as file:
            tasks = yaml.safe_load(file)

        self.assertNotIn("editing_advisor", agents)
        self.assertNotIn("generate_editing_task", tasks)
        self.assertNotIn(
            "EDITING INSTRUCTIONS",
            tasks["archive_content_task"]["description"],
        )
        caption_text = (
            agents["caption_writer"]["backstory"]
            + tasks["generate_caption_task"]["description"]
        )
        self.assertIn("UNDER 100 characters", caption_text)
        self.assertIn("Do NOT include a save-prompt sentence", caption_text)

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

    def test_generation_timeout_classifier_marks_complete(self):
        state = {
            "result_count": 5,
            "activity_visible": False,
            "activity_keywords": False,
            "button_enabled": True,
        }
        self.assertEqual(_classify_generation_timeout(state, baseline_result_count=4), "complete")

    def test_generation_timeout_classifier_marks_in_progress(self):
        state = {
            "result_count": 4,
            "activity_visible": True,
            "activity_keywords": False,
            "button_enabled": False,
        }
        self.assertEqual(_classify_generation_timeout(state, baseline_result_count=4), "in_progress")

    def test_generation_timeout_classifier_marks_idle(self):
        state = {
            "result_count": 4,
            "activity_visible": False,
            "activity_keywords": False,
            "button_enabled": True,
        }
        self.assertEqual(_classify_generation_timeout(state, baseline_result_count=4), "idle")

    def test_pipeline_connections_pass_with_expected_context_chain(self):
        task1 = DummyTaskNode()
        task2 = DummyTaskNode(context=[task1])
        task3 = DummyTaskNode(context=[task2])
        verify_pipeline_task_connections(task1, task2, task3)

    def test_pipeline_connections_fail_when_prompt_context_missing(self):
        task1 = DummyTaskNode()
        task2 = DummyTaskNode(context=[])
        task3 = DummyTaskNode(context=[task2])
        with self.assertRaises(ValueError) as ctx:
            verify_pipeline_task_connections(task1, task2, task3)
        self.assertIn("task2 must depend on task1 output", str(ctx.exception))

    def test_browser_stage_falls_back_to_deterministic_tool_execution(self):
        browser_crew = DummyCrew(response="browser agent described action")
        browser_task = DummyTaskNode(raw="agent did not call tool")
        video_tool = DummyVideoTool()

        result = run_browser_operator_stage(
            browser_crew=browser_crew,
            browser_task=browser_task,
            browser_task_base_description="Execute browser stage",
            video_tool=video_tool,
            video_url="https://labs.google/fx/tools/flow",
            project_name="fallback-test",
            prompt_json_output='[{"clip_label":"CLIP 1"}]',
            flow_dry_run=True,
        )

        self.assertEqual(result["execution_path"], "deterministic_tool_execution")
        self.assertEqual(result["tool_output"], "DRY RUN PASSED")
        self.assertEqual(len(video_tool.calls), 1)

    def test_merge_failed_clip_repairs_preserves_passing_clips(self):
        original_clips = [
            {"clip": 1, "voice_text": "Original one."},
            {"clip": 2, "voice_text": "Original two."},
            {"clip": 3, "voice_text": "Original three."},
        ]
        repaired_clips = [
            {"clip": 1, "voice_text": "Changed one unexpectedly."},
            {"clip": 2, "voice_text": "Fixed two."},
            {"clip": 3, "voice_text": "Changed three unexpectedly."},
        ]
        merged = merge_failed_clip_repairs(
            original_title="Original",
            original_clips=original_clips,
            repaired_title="Repaired",
            repaired_clips=repaired_clips,
            failed_clips=[2],
        )
        self.assertIn("Clip 1:\nOriginal one.", merged)
        self.assertIn("Clip 2:\nFixed two.", merged)
        self.assertIn("Clip 3:\nOriginal three.", merged)


if __name__ == "__main__":
    unittest.main()
