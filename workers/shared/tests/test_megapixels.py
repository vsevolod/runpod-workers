"""Common megapixel policy: hardware selection and request compatibility."""

import importlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[2]))


class MegapixelPolicyTests(unittest.TestCase):
    def test_vram_probe_reads_total_memory_of_the_active_device(self):
        policy = importlib.import_module("shared.megapixels")
        def properties(index):
            self.assertEqual(index, 1)
            return SimpleNamespace(total_memory=48 * 1024**3)
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, current_device=lambda: 1,
                                                     get_device_properties=properties))
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(policy.detect_vram_gib(), 48.)

    def test_max_uses_the_selected_model_and_nominal_vram_bucket(self):
        policy = importlib.import_module("shared.megapixels")
        with patch.dict(policy.MEGAPIXEL_LIMITS, {
            "qwen_image_21.base": {0: 1., 24: 2., 48: 3.},
            "qwen_image_21.turbo": {0: 1., 24: 1.5},
        }), patch.object(policy, "detect_vram_gib", return_value=23.7):
            self.assertEqual(policy.resolve_megapixels("max", "qwen_image_21.base").megapixels, 2.)
            self.assertEqual(policy.resolve_megapixels("max", "qwen_image_21.turbo").megapixels, 1.5)
        with patch.object(policy, "detect_vram_gib", return_value=47.5), patch.dict(
            policy.MEGAPIXEL_LIMITS, {"qwen_image_21.base": {0: 1., 24: 2., 48: 3.}}
        ):
            self.assertEqual(policy.resolve_megapixels("max", "qwen_image_21.base").megapixels, 3.)

    def test_explicit_number_bypasses_gpu_detection_and_table(self):
        policy = importlib.import_module("shared.megapixels")
        with patch.object(policy, "detect_vram_gib", side_effect=AssertionError("GPU must not be queried")):
            budget = policy.resolve_megapixels(.5, "qwen_image_21.base")
        self.assertEqual(budget.megapixels, .5)
        self.assertIsNone(budget.vram_gib)

    def test_l40s_cuda_memory_selects_four_megapixels_for_calibrated_profiles(self):
        policy = importlib.import_module("shared.megapixels")
        # Actual CUDA total from the 2026-10-18 L40S 48 GB benchmark.
        with patch.object(policy, "detect_vram_gib", return_value=47_677_177_856 / 1024**3):
            for model in ("krea2.fp8", "krea2.int8_convrot",
                          "qwen_image_21.base", "qwen_image_21.turbo"):
                with self.subTest(model=model):
                    budget = policy.resolve_megapixels("max", model)
                    self.assertEqual(budget.megapixels, 4.)
                    self.assertEqual(budget.vram_gib, 44.40283203125)

    def test_capacity_conversion_does_not_promote_smaller_cards_to_48gb(self):
        policy = importlib.import_module("shared.megapixels")
        with patch.dict(policy.MEGAPIXEL_LIMITS, {"krea2.fp8": {0: 1., 32: 2., 48: 4.}}):
            for vram_gib in (31.5, 32., 40.):
                with self.subTest(vram_gib=vram_gib), patch.object(
                    policy, "detect_vram_gib", return_value=vram_gib
                ):
                    self.assertEqual(policy.resolve_megapixels("max", "krea2.fp8").megapixels, 2.)

    def test_missing_cuda_and_unknown_models_use_one_megapixel(self):
        policy = importlib.import_module("shared.megapixels")
        with patch.object(policy, "detect_vram_gib", return_value=None):
            self.assertEqual(policy.resolve_megapixels("max", "krea2.fp8").megapixels, 1.)
        with patch.object(policy, "detect_vram_gib", return_value=80.):
            self.assertEqual(policy.resolve_megapixels("max", "unknown").megapixels, 1.)

    def test_both_profiles_accept_top_level_max_default_and_explicit_number(self):
        from shared.api import KREA_API, QWEN_API, parse_job_input
        for profile in (KREA_API, QWEN_API):
            for fields, expected in (({}, "max"), ({"reference_megapixels": "max"}, "max"),
                                     ({"reference_megapixels": .5}, .5)):
                parsed = parse_job_input({"input": {"prompt": "fox", **fields}}, profile)
                self.assertEqual(parsed.values["reference_megapixels"], expected)
                self.assertNotIn("reference_megapixels", parsed.values["model_options"])

    def test_legacy_nested_budget_is_normalized_without_mutating_input_and_conflicts_fail(self):
        from shared.api import QWEN_API, RequestError, parse_job_input
        job = {"input": {"prompt": "fox", "model_options": {"reference_megapixels": .5}}}
        parsed = parse_job_input(job, QWEN_API)
        self.assertEqual(parsed.values["reference_megapixels"], .5)
        self.assertEqual(job["input"]["model_options"], {"reference_megapixels": .5})
        with self.assertRaisesRegex(RequestError, "conflict"):
            parse_job_input({"input": {**job["input"], "reference_megapixels": "max"}}, QWEN_API)

    def test_invalid_common_budget_is_rejected(self):
        from shared.api import KREA_API, QWEN_API, RequestError, parse_job_input
        for profile in (KREA_API, QWEN_API):
            for value in (True, None, "1", "MAX", 0, -1, float("inf"), float("nan"), 4.4):
                with self.subTest(profile=profile.name, value=value), self.assertRaises(RequestError):
                    parse_job_input({"input": {"prompt": "fox", "reference_megapixels": value}}, profile)

    def test_invalid_duplicate_budget_cannot_be_hidden_by_legacy_numeric_value(self):
        from shared.api import QWEN_API, RequestError, parse_job_input
        for top, nested in ((True, 1.), (1., True), ("1", 1.)):
            with self.subTest(top=top, nested=nested), self.assertRaises(RequestError):
                parse_job_input({"input": {"prompt": "fox", "reference_megapixels": top,
                                           "model_options": {"reference_megapixels": nested}}}, QWEN_API)


if __name__ == "__main__":
    unittest.main()
