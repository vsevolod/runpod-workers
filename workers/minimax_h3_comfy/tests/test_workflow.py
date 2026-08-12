"""Strict inject tests — template must exist; exact node ids from PINS.md."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workflow import (  # noqa: E402
    DURATION_KEY,
    DURATION_NODE,
    FIRST_FRAME_KEY,
    FIRST_LOAD_NODE,
    LAST_FRAME_KEY,
    LAST_LOAD_NODE,
    PROMPT_KEY,
    PROMPT_NODE,
    SEED_KEY,
    SEED_NODE,
    UNET_NAME_KEY,
    UNET_NODE,
    inject_product,
    load_workflow,
    template_path,
    validate_canvas,
)


class TestWorkflowInject(unittest.TestCase):
    def test_template_exists(self):
        self.assertTrue(template_path().is_file(), f"missing {template_path()}")

    def test_inject_sets_prompt_and_seed_exactly(self):
        wf = load_workflow()
        out = inject_product(
            wf, prompt="hello", width=864, height=480, duration=5.0, seed=42
        )
        self.assertEqual(out["104"]["inputs"]["prompt"], "hello")
        self.assertEqual(out["15"]["inputs"]["noise_seed"], 42)
        self.assertEqual(out["104"]["inputs"]["width"], 864)
        self.assertEqual(out["104"]["inputs"]["height"], 480)
        self.assertEqual(out["111"]["inputs"]["value"], 5.0)

    def test_inject_does_not_change_unrelated_nodes(self):
        wf = load_workflow()
        unet_before = copy.deepcopy(wf["6"]["inputs"]["unet_name"])
        clip_before = copy.deepcopy(wf["13"]["inputs"]["clip_name"])
        save_before = copy.deepcopy(wf["92"]["inputs"])
        out = inject_product(
            wf, prompt="x", width=864, height=480, duration=5.0, seed=1
        )
        self.assertEqual(
            out["6"]["inputs"]["unet_name"],
            unet_before,
        )
        self.assertEqual(out["13"]["inputs"]["clip_name"], clip_before)
        self.assertEqual(out["92"]["inputs"], save_before)
        # original workflow not mutated
        self.assertEqual(wf["104"]["inputs"]["prompt"], load_workflow()["104"]["inputs"]["prompt"])

    def test_inject_missing_node_raises(self):
        wf = load_workflow()
        del wf["104"]
        with self.assertRaises(KeyError):
            inject_product(
                wf, prompt="x", width=864, height=480, duration=5.0, seed=1
            )

    def test_inject_missing_key_raises(self):
        wf = load_workflow()
        del wf["15"]["inputs"]["noise_seed"]
        with self.assertRaises(KeyError):
            inject_product(
                wf, prompt="x", width=864, height=480, duration=5.0, seed=1
            )

    def test_validate_canvas_rejects_non_multiple(self):
        with self.assertRaises(ValueError):
            validate_canvas(865, 480)

    def test_validate_canvas_rejects_too_large(self):
        with self.assertRaises(ValueError):
            validate_canvas(1920, 1088)

    def test_constants_match_pins_literals(self):
        self.assertEqual(PROMPT_NODE, "104")
        self.assertEqual(PROMPT_KEY, "prompt")
        self.assertEqual(DURATION_NODE, "111")
        self.assertEqual(DURATION_KEY, "value")
        self.assertEqual(SEED_NODE, "15")
        self.assertEqual(SEED_KEY, "noise_seed")
        self.assertEqual(UNET_NODE, "6")
        self.assertEqual(UNET_NAME_KEY, "unet_name")
        self.assertEqual(FIRST_LOAD_NODE, "200")
        self.assertEqual(LAST_LOAD_NODE, "201")
        self.assertEqual(FIRST_FRAME_KEY, "first_frame")
        self.assertEqual(LAST_FRAME_KEY, "last_frame")

    def test_inject_t2v_has_no_frame_links_or_loadimage(self):
        wf = load_workflow()
        out = inject_product(
            wf, prompt="hi", width=864, height=480, duration=2.0, seed=1
        )
        self.assertNotIn(FIRST_FRAME_KEY, out["104"]["inputs"])
        self.assertNotIn(LAST_FRAME_KEY, out["104"]["inputs"])
        self.assertNotIn(FIRST_LOAD_NODE, out)
        self.assertNotIn(LAST_LOAD_NODE, out)

    def test_inject_first_frame_only(self):
        wf = load_workflow()
        out = inject_product(
            wf,
            prompt="hi",
            width=864,
            height=480,
            duration=2.0,
            seed=1,
            first_image_name="job_first.png",
        )
        self.assertEqual(out[FIRST_LOAD_NODE]["class_type"], "LoadImage")
        self.assertEqual(out[FIRST_LOAD_NODE]["inputs"]["image"], "job_first.png")
        self.assertEqual(out["104"]["inputs"][FIRST_FRAME_KEY], [FIRST_LOAD_NODE, 0])
        self.assertNotIn(LAST_FRAME_KEY, out["104"]["inputs"])
        self.assertNotIn(LAST_LOAD_NODE, out)

    def test_inject_first_and_last(self):
        wf = load_workflow()
        out = inject_product(
            wf,
            prompt="hi",
            width=864,
            height=480,
            duration=2.0,
            seed=1,
            first_image_name="a.png",
            last_image_name="b.png",
        )
        self.assertEqual(out["104"]["inputs"][FIRST_FRAME_KEY], [FIRST_LOAD_NODE, 0])
        self.assertEqual(out["104"]["inputs"][LAST_FRAME_KEY], [LAST_LOAD_NODE, 0])
        self.assertEqual(out[LAST_LOAD_NODE]["inputs"]["image"], "b.png")

    def test_inject_last_without_first_raises(self):
        wf = load_workflow()
        with self.assertRaises(ValueError):
            inject_product(
                wf,
                prompt="hi",
                width=864,
                height=480,
                duration=2.0,
                seed=1,
                last_image_name="b.png",
            )


if __name__ == "__main__":
    unittest.main()

