"""Shared request contract: compatibility, model capabilities and strict inputs."""

import importlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[2]))


class APIContractTests(unittest.TestCase):
    def setUp(self):
        try:
            self.api = importlib.import_module("shared.api")
        except ModuleNotFoundError:
            self.fail("The shared worker API is not implemented")

    def parse(self, profile, **fields):
        return self.api.parse_job_input({"input": {"prompt": "fox", **fields}}, profile)

    def test_qwen_nested_variant_changes_default_steps_without_mutating_input(self):
        raw = {"input": {"prompt": "fox", "model_options": {"variant": "base"}}}
        parsed = self.api.parse_job_input(raw, self.api.QWEN_API)
        self.assertEqual(parsed.values["model_variant"], "base")
        self.assertEqual(parsed.values["num_inference_steps"], 40)
        self.assertNotIn("num_inference_steps", parsed.raw_keys)
        self.assertEqual(raw, {"input": {"prompt": "fox", "model_options": {"variant": "base"}}})

    def test_legacy_and_nested_parameters_have_identical_effective_values(self):
        cases = [
            ("QWEN_API", {"model_variant": "base", "output_resolution": 2048}, {"variant": "base", "output_resolution": 2048}),
            ("KREA_API", {"mu": .9, "grounding_px": 0, "ref_boost": 4., "fit_mode": "crop"},
             {"mu": .9, "grounding_px": 0, "ref_boost": 4., "fit_mode": "crop"}),
        ]
        for name, legacy, nested in cases:
            profile = getattr(self.api, name)
            old = self.parse(profile, **legacy)
            new = self.parse(profile, model_options=nested)
            self.assertEqual(old.values, new.values)
            self.assertEqual(old.raw_keys, new.raw_keys)
            same = self.parse(profile, model_options=nested, **legacy)
            self.assertEqual(old.values, same.values)

    def test_conflicting_or_foreign_options_are_rejected(self):
        for profile, fields, message in [
            ("QWEN_API", {"model_variant": "turbo", "model_options": {"variant": "base"}}, "conflict"),
            ("KREA_API", {"mu": 1.15, "model_options": {"mu": .9}}, "conflict"),
            ("QWEN_API", {"model_options": {"mu": .9}}, "mu"),
            ("KREA_API", {"model_options": {"variant": "base"}}, "variant"),
            ("QWEN_API", {"model_options": {"varaint": "base"}}, "varaint"),
            ("KREA_API", {"model_options": None}, "model_options"),
            ("KREA_API", {"unknown": 1}, "unknown"),
        ]:
            with self.subTest(fields=fields), self.assertRaisesRegex(self.api.RequestError, message):
                self.parse(getattr(self.api, profile), **fields)

    def test_capabilities_and_defaults_are_selected_by_model(self):
        for name, steps, guidance in [("KREA_API", 8, 0.), ("QWEN_API", 8, 1.)]:
            parsed = self.parse(getattr(self.api, name))
            self.assertEqual(parsed.values["num_inference_steps"], steps)
            self.assertEqual(parsed.values["guidance_scale"], guidance)
            self.assertEqual(parsed.values["sampler"], "euler")
        for sampler in ("euler", "heun", "er_sde"):
            self.assertEqual(self.parse(self.api.QWEN_API, sampler=sampler).values["sampler"], sampler)
        for sampler in ("heun", "er_sde"):
            with self.assertRaisesRegex(self.api.RequestError, "sampler"):
                self.parse(self.api.KREA_API, sampler=sampler)

    def test_validation_rejects_bad_values_instead_of_defaulting_or_starting_inference(self):
        for profile, fields in [
            ("QWEN_API", {"num_inference_steps": 201}),
            ("KREA_API", {"num_inference_steps": 65}),
            ("KREA_API", {"num_inference_steps": 0}),
            ("KREA_API", {"num_inference_steps": True}),
            ("QWEN_API", {"guidance_scale": float("nan")}),
            ("KREA_API", {"guidance_scale": float("inf")}),
            ("QWEN_API", {"seed": True}),
            ("QWEN_API", {"images": "not a list"}),
            ("KREA_API", {"model_options": {"grounding_px": 2}}),
            ("KREA_API", {"model_options": {"ref_boost": -1}}),
            ("KREA_API", {"model_options": {"mu": float("nan")}}),
            ("QWEN_API", {"model_options": {"output_resolution": 1000}}),
        ]:
            with self.subTest(fields=fields), self.assertRaises(self.api.RequestError):
                self.parse(getattr(self.api, profile), **fields)
        self.assertEqual(self.parse(self.api.KREA_API, guidance_scale=3).values["guidance_scale"], 3.)
        self.assertEqual(self.parse(self.api.QWEN_API, seed=0).values["seed"], 0)

    def test_invalid_job_envelopes_are_client_errors(self):
        for job in (None, [], {}, {"input": None}, {"input": []}, {"input": "x"}):
            with self.subTest(job=job), self.assertRaises(self.api.RequestError):
                self.api.parse_job_input(job, self.api.QWEN_API)


if __name__ == "__main__":
    unittest.main()
