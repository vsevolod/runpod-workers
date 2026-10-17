"""Krea's thin API adapter preserves options and LoRA types at inference."""

import base64
import importlib.util
from io import BytesIO
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

WORKER = Path(__file__).parents[1]
sys.path.insert(0, str(WORKER.parent))


class AdapterError(ValueError):
    pass


class Selection:
    def __init__(self, item):
        self.name = item["name"]
        self.strength = item.get("strength", 1.)
        self.type = item.get("type", "lora")
    def as_dict(self):
        return vars(self)


class Pipeline:
    def __init__(self):
        self.calls = []
        self.loras = SimpleNamespace(normalize=lambda items: tuple(Selection(item) for item in items or []))
    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return [Image.new("RGB", (1024, 1024))]
    def edit(self, **kwargs):
        self.calls.append(("edit", kwargs))
        return [Image.new("RGB", (512, 768))]


class KreaAPITests(unittest.TestCase):
    def setUp(self):
        self.pipe = Pipeline()
        package = ModuleType("krea2_infer")
        package.__path__ = [str(WORKER / "krea2_infer")]
        package.load_pipeline = lambda **kwargs: self.pipe
        torch = SimpleNamespace(
            inference_mode=lambda: lambda function: function,
            cuda=SimpleNamespace(OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}), is_available=lambda: False),
        )
        modules = {
            "krea2_infer": package,
            "krea2_infer.lora": SimpleNamespace(LoRAError=AdapterError),
            "torch": torch,
            "runpod": SimpleNamespace(serverless=SimpleNamespace(start=lambda _: None)),
            "runpod.serverless.utils": SimpleNamespace(rp_cleanup=None, rp_upload=None),
            "runpod.serverless.utils.rp_validator": SimpleNamespace(validate=lambda value, rules: {"validated_input": {
                **{key: rule.get("default") for key, rule in rules.items()}, **value,
            }}),
        }
        schema_spec = importlib.util.spec_from_file_location("_krea_api_schema", WORKER / "schemas.py")
        schema = importlib.util.module_from_spec(schema_spec); schema_spec.loader.exec_module(schema)
        modules["schemas"] = schema
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start(); self.addCleanup(self.patch.stop)
        spec = importlib.util.spec_from_file_location("_krea_api_handler", WORKER / "handler.py")
        self.handler = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.handler)
        self.handler._save_and_upload_images = lambda *args: ["data:image/png;base64,test"]

    def test_nested_and_legacy_edit_options_forward_identical_values_and_keep_weight_diff(self):
        buffer = BytesIO(); Image.new("RGB", (64, 96)).save(buffer, format="PNG")
        options = {"mu": .9, "grounding_px": 0, "ref_boost": 4., "fit_mode": "crop"}
        for fields in ({"model_options": options}, options):
            output = self.handler.generate_image({"input": {
                "type": "image_edit", "prompt": "fox", "seed": 0,
                "num_inference_steps": 25, "guidance_scale": 2.5, "sampler": "euler",
                "images": [base64.b64encode(buffer.getvalue()).decode()],
                "loras": [{"name": "identity", "type": "weight_diff", "strength": 4.}], **fields,
            }})
            self.assertNotIn("error", output)
            self.assertEqual(output["model_options"], options)
            self.assertEqual(output["ref_boost"], 4.)
            self.assertEqual((output["width"], output["height"]), (512, 768))
            self.assertEqual(output["num_inference_steps"], 25)
            self.assertEqual(output["guidance_scale"], 2.5)
            self.assertEqual(output["sampler"], "euler")
            kwargs = self.pipe.calls[-1][1]
            self.assertEqual(kwargs["steps"], 25)
            self.assertEqual(kwargs["guidance"], 2.5)
            self.assertEqual(kwargs["seed"], 0)
            self.assertIsNone(kwargs["width"])
            self.assertEqual(kwargs["mu"], .9)
            self.assertEqual(kwargs["loras"][0].type, "weight_diff")
            self.assertEqual(kwargs["loras"][0].strength, 4.)
            self.assertEqual(output["loras"][0]["type"], "weight_diff")

    def test_invalid_options_or_sampler_are_rejected_before_inference(self):
        for fields in ({"sampler": "heun"}, {"sampler": "er_sde"},
                       {"model_options": {"variant": "base"}},
                       {"mu": 1.15, "model_options": {"mu": .9}},
                       {"num_inference_steps": 0}):
            output = self.handler.generate_image({"input": {"prompt": "fox", **fields}})
            self.assertIn("error", output)
            self.assertNotIn("refresh_worker", output)
        self.assertEqual(self.pipe.calls, [])

    def test_legacy_generation_keeps_defaults_and_reports_them(self):
        output = self.handler.generate_image({"input": {"prompt": "fox", "seed": 7}})
        self.assertNotIn("error", output)
        self.assertEqual(output["num_inference_steps"], 8)
        self.assertEqual(output["guidance_scale"], 0.)
        self.assertEqual(output["seed"], 7)
        self.assertEqual(output["model_options"]["mu"], 1.15)
        self.assertEqual(self.pipe.calls[-1][1]["steps"], 8)

    def test_null_mu_keeps_auto_generation_and_resolves_to_fixed_edit_shift(self):
        output = self.handler.generate_image({"input": {"prompt": "fox", "model_options": {"mu": None}}})
        self.assertNotIn("error", output)
        self.assertIsNone(output["model_options"]["mu"])
        self.assertIsNone(self.pipe.calls[-1][1]["mu"])
        buffer = BytesIO(); Image.new("RGB", (64, 96)).save(buffer, format="PNG")
        output = self.handler.generate_image({"input": {
            "prompt": "fox", "type": "image_edit", "model_options": {"mu": None},
            "images": [base64.b64encode(buffer.getvalue()).decode()],
        }})
        self.assertNotIn("error", output)
        self.assertEqual(output["model_options"]["mu"], 1.15)
        self.assertEqual(self.pipe.calls[-1][1]["mu"], 1.15)


if __name__ == "__main__":
    unittest.main()
