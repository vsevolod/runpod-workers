import os
import base64
import importlib.util
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
from contextlib import nullcontext

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer.lora import LoRACatalog
from qwen_image_21_infer.pipeline import QwenImage21Runtime
from test_lora import Pipeline

WORKER = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Generator:
    def __init__(self, device):
        self.device = device

    def manual_seed(self, seed):
        self.seed = seed
        return self


class OutOfMemoryError(RuntimeError):
    pass


class InferencePipeline(Pipeline):
    def __init__(self):
        super().__init__()
        self.vae = object()
        self.calls = []
        self.failure = None
        self.scheduler = object()
        self.seen_samplers = []

    def remove_all_hooks(self):
        pass

    def __call__(self, **kwargs):
        self.calls.append((kwargs, list(self.active)))
        self.seen_samplers.append(self.scheduler)
        if self.failure:
            raise self.failure
        return SimpleNamespace(images=[Image.new("RGB", (32, 32))])


class LoRAAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cache_env = patch.dict(os.environ, {"LORA_CACHE_DIR": str(Path(self.tmp.name) / "local"), "LORA_DIR": self.tmp.name})
        cache_env.start()
        self.addCleanup(cache_env.stop)
        (Path(self.tmp.name) / "style.safetensors").write_bytes(b"weights")
        self.pipe = InferencePipeline()
        torch = SimpleNamespace(
            inference_mode=lambda: lambda function: function,
            Generator=Generator,
            cuda=SimpleNamespace(
                is_available=lambda: False, OutOfMemoryError=OutOfMemoryError
            ),
        )
        schema = load_module("_qwen_lora_schema", WORKER / "schemas.py")

        def validate(value, rules):
            defaults = {key: entry.get("default") for key, entry in rules.items()}
            return {"validated_input": {**defaults, **value}}

        modules = {
            "torch": torch,
            "runpod": SimpleNamespace(serverless=SimpleNamespace(start=Mock())),
            "runpod.serverless.utils": SimpleNamespace(
                rp_cleanup=Mock(), rp_upload=Mock()
            ),
            "runpod.serverless.utils.rp_validator": SimpleNamespace(validate=validate),
            "schemas": schema,
        }
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, modules).start()
        patch("qwen_image_21_infer.pipeline.make_scheduler", side_effect=lambda template, sampler, schedule, **kwargs: (sampler, schedule)).start()
        self.handler = load_module("_qwen_lora_handler", WORKER / "handler.py")
        from qwen_image_21_infer.model_manager import ModelManager
        self.handler.MODELS = ModelManager(loader=lambda **kwargs: QwenImage21Runtime(self.pipe, LoRACatalog.scan(self.tmp.name)), prepare=lambda variant: nullcontext())
        self.handler._save_and_upload_images = lambda images, job_id: [
            "data:image/png;base64,test"
        ]

    def test_generation_and_edit_forward_loras_and_report_effective_selection(self):
        buffer = BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue()).decode()
        for mode in ["image_generate", "image_edit"]:
            payload = {
                "prompt": "fox",
                "type": mode,
                "seed": 42,
                "loras": [{"name": "style", "strength": 0.8}],
            }
            if mode == "image_edit":
                payload["images"] = [image]
            output = self.handler.generate_image({"input": payload})
            self.assertNotIn("error", output)
            self.assertEqual(
                output["loras"], [{"name": "style", "strength": 0.8, "type": "lora"}]
            )
            kwargs, adapters = self.pipe.calls[-1]
            self.assertNotIn("loras", kwargs)
            self.assertEqual(kwargs["generator"].seed, 42)
            self.assertEqual(output["seed"], 42)
            self.assertEqual(adapters, [("request_lora_0", 0.8)])
            self.assertEqual(output["num_inference_steps"], 8)
            self.assertNotIn("sigmas", kwargs)
            self.assertEqual(self.pipe.loaded, set())
        self.assertIn("image", self.pipe.calls[-1][0])

    def test_omitted_and_null_seed_generate_new_seeds_and_report_them(self):
        buffer = BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue()).decode()
        for mode in ["image_generate", "image_edit"]:
            with self.subTest(mode=mode):
                payload = {"prompt": "fox", "type": mode}
                if mode == "image_edit":
                    payload["images"] = [image]
                with patch.object(self.handler.os, "urandom", side_effect=[b"\x00\x00\x00\x07", b"\x00\x00\x00\x09"]) as entropy:
                    for expected_seed in [7, 9]:
                        output = self.handler.generate_image({"input": payload})
                        self.assertNotIn("error", output)
                        self.assertEqual(output["seed"], expected_seed)
                        self.assertEqual(self.pipe.calls[-1][0]["generator"].seed, expected_seed)
                        payload["seed"] = None
                    self.assertEqual(entropy.call_args_list, [call(4), call(4)])

    def test_explicit_zero_seed_does_not_request_randomness(self):
        with patch.object(self.handler.os, "urandom") as entropy:
            output = self.handler.generate_image({"input": {"prompt": "fox", "seed": 0}})
        entropy.assert_not_called()
        self.assertEqual(output["seed"], 0)
        self.assertEqual(self.pipe.calls[-1][0]["generator"].seed, 0)

    def test_omitted_loras_preserve_existing_api(self):
        output = self.handler.generate_image({"input": {"prompt": "fox"}})
        self.assertEqual(output["loras"], [])
        self.assertEqual(self.pipe.calls[-1][1], [])

    def test_base_edit_uses_forty_steps_native_refs_and_reports_selected_model(self):
        buffer = BytesIO()
        Image.new("RGB", (768, 512)).save(buffer, format="PNG")
        output = self.handler.generate_image({"input": {
            "type": "image_edit", "model_variant": "base", "prompt": "fox", "seed": 7,
            "images": [base64.b64encode(buffer.getvalue()).decode()],
            "loras": [{"name": "style", "strength": .8}],
        }})
        self.assertNotIn("error", output)
        kwargs, adapters = self.pipe.calls[-1]
        self.assertEqual(kwargs["num_inference_steps"], 40)
        self.assertEqual(len(kwargs["sigmas"]), 40)
        self.assertEqual(kwargs["reference_resolution"], 0)
        self.assertEqual((kwargs["width"], kwargs["height"]), (1216, 800))
        self.assertEqual(kwargs["image"][0].size, (768, 512))
        self.assertEqual(adapters, [("request_lora_0", .8)])
        self.assertEqual(output["model_variant"], "base")
        self.assertEqual(output["model_id"], "Qwen/Qwen-Image-2.1")
        self.assertEqual(output["scheduler"], "simple")
        self.assertEqual(output["seed"], 7)

    def test_nested_model_options_select_base_and_preserve_legacy_response_fields(self):
        output = self.handler.generate_image({"input": {
            "prompt": "fox", "model_options": {"variant": "base", "output_resolution": 2048},
        }})
        self.assertNotIn("error", output)
        self.assertEqual(output["model_variant"], "base")
        self.assertEqual(output["num_inference_steps"], 40)
        self.assertEqual(output["model_options"], {"variant": "base", "output_resolution": 2048})
        self.assertEqual(output["reference_megapixels"], "max")
        self.assertEqual(output["resolved_reference_megapixels"], 1.)
        self.assertEqual(self.pipe.calls[-1][0]["num_inference_steps"], 40)

    def test_invalid_nested_options_do_not_load_or_run_the_model(self):
        for fields in ({"model_options": {"mu": 1.}},
                       {"model_variant": "turbo", "model_options": {"variant": "base"}}):
            output = self.handler.generate_image({"input": {"prompt": "fox", **fields}})
            self.assertIn("error", output)
        self.assertEqual(self.pipe.calls, [])

    def test_auto_resolution_in_model_options_stays_reusable_for_megapixel_edits(self):
        buffer = BytesIO()
        Image.new("RGB", (800, 1184)).save(buffer, format="PNG")
        payload = {"type": "image_edit", "prompt": "fox", "images": [base64.b64encode(buffer.getvalue()).decode()]}
        output = self.handler.generate_image({"input": payload})
        self.assertNotIn("error", output)
        self.assertIsNone(output["model_options"]["output_resolution"])
        repeated = self.handler.generate_image({"input": {**payload, "model_options": output["model_options"],
                                                         "reference_megapixels": output["reference_megapixels"]}})
        self.assertNotIn("error", repeated)
        self.assertEqual(self.pipe.calls[-1][0]["width"], 800)
        self.assertEqual(self.pipe.calls[-1][0]["height"], 1216)

    def test_legacy_reference_budget_is_applied_and_returned_at_top_level(self):
        buffer = BytesIO()
        Image.new("RGBA", (2048, 2048), (10, 20, 30, 80)).save(buffer, format="PNG")
        payload = {"type": "image_edit", "prompt": "fox", "width": 1024, "height": 1024,
                   "images": [base64.b64encode(buffer.getvalue()).decode()] * 2,
                   "model_options": {"variant": "base", "reference_megapixels": .5}}
        output = self.handler.generate_image({"input": payload})
        self.assertNotIn("error", output)
        self.assertEqual(output["reference_megapixels"], .5)
        self.assertEqual(output["resolved_reference_megapixels"], .5)
        self.assertNotIn("reference_megapixels", output["model_options"])
        kwargs = self.pipe.calls[-1][0]
        self.assertEqual([image.size for image in kwargs["image"]], [(480, 480)] * 2)
        self.assertEqual((kwargs["width"], kwargs["height"]), (1024, 1024))
        repeated = self.handler.generate_image({"input": {**payload, "model_options": output["model_options"],
                                                         "reference_megapixels": output["reference_megapixels"]}})
        self.assertNotIn("error", repeated)
        self.assertEqual([image.size for image in self.pipe.calls[-1][0]["image"]], [(480, 480)] * 2)

    def test_invalid_variant_or_step_count_does_not_prepare_or_load_a_model(self):
        with patch.object(self.handler.MODELS, "_loader") as load, patch.object(self.handler.MODELS, "_prepare") as prepare:
            for payload in [{"model_variant": "other"}, {"num_inference_steps": 0}, {"sampler": "unknown"}]:
                output = self.handler.generate_image({"input": {"prompt": "fox", **payload}})
                self.assertIn("error", output)
            load.assert_not_called()
            prepare.assert_not_called()

    def test_sampling_overrides_reach_generation_and_edit_and_are_reported(self):
        buffer = BytesIO()
        Image.new("RGB", (256, 256)).save(buffer, format="PNG")
        for mode in ("image_generate", "image_edit"):
            for variant in ("base", "turbo"):
                for sampler in ("euler", "heun", "er_sde"):
                    payload = {"prompt": "fox", "type": mode, "model_variant": variant,
                               "num_inference_steps": 25, "guidance_scale": 3.5, "sampler": sampler}
                    if mode == "image_edit":
                        payload["images"] = [base64.b64encode(buffer.getvalue()).decode()]
                    output = self.handler.generate_image({"input": payload})
                    self.assertNotIn("error", output)
                    kwargs = self.pipe.calls[-1][0]
                    self.assertEqual(kwargs["num_inference_steps"], 25)
                    self.assertEqual(len(kwargs["sigmas"]), 25)
                    self.assertEqual(kwargs["true_cfg_scale"], 3.5)
                    self.assertEqual(kwargs["negative_prompt"], "")
                    self.assertEqual(output["num_inference_steps"], 25)
                    self.assertEqual(output["guidance_scale"], 3.5)
                    self.assertEqual(output["sampler"], sampler)
                    self.assertEqual(output["scheduler"], "simple")
                    self.assertEqual(self.pipe.seen_samplers[-1], (sampler, "simple"))

    def test_invalid_name_and_incompatible_weights_do_not_refresh_worker(self):
        for name in ["missing", "style"]:
            self.pipe.state = {}
            output = self.handler.generate_image(
                {"input": {"prompt": "fox", "loras": [{"name": name}]}}
            )
            self.assertIn("error", output)
            self.assertNotIn("refresh_worker", output)
            self.assertEqual(self.pipe.calls, [])
        self.assertEqual(self.pipe.loaded, set())

    def test_oom_keeps_restart_behavior_and_restores_base(self):
        self.pipe.failure = OutOfMemoryError("OOM")
        scheduler = self.pipe.scheduler
        output = self.handler.generate_image(
            {"input": {"prompt": "fox", "loras": [{"name": "style"}]}}
        )
        self.assertTrue(output["refresh_worker"])
        self.assertEqual(self.pipe.loaded, set())
        self.assertIs(self.pipe.scheduler, scheduler)

class ClientLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = load_module(
            "_qwen_lora_client", WORKER.parents[1] / "scripts" / "qwen_image_21.py"
        )

    def test_cli_uses_variant_specific_schedules(self):
        parser = self.client._build_parser()
        self.assertEqual(self.client._request_steps(parser.parse_args([])), 8)
        self.assertEqual(self.client._request_steps(parser.parse_args(["--model-variant", "base"])), 40)
        self.assertEqual(self.client._request_steps(parser.parse_args(["--model-variant", "base", "--steps", "25"])), 25)
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--steps", "0"])

    def test_cli_forwards_steps_cfg_and_sampler_for_both_modes(self):
        for edit in (False, True):
            command = ["--prompt", "fox", "--steps", "25", "--cfg", "3.5", "--sampler", "er_sde"]
            if edit:
                command += ["--image", "ref.png"]
            args = self.client._build_parser().parse_args(command)
            with patch.object(self.client, "_submit_job", return_value="job") as submit, patch.object(
                self.client, "_poll_job", return_value={}
            ), patch.object(self.client, "_save_result", return_value=0), patch.object(self.client, "_image_to_data_url", return_value="base64"):
                method = self.client._run_edit if edit else self.client._run_generate
                self.assertEqual(method(args, endpoint_id="same-endpoint", api_key="key"), 0)
            payload = submit.call_args.args[2]
            self.assertEqual(payload["num_inference_steps"], 25)
            self.assertEqual(payload["guidance_scale"], 3.5)
            self.assertEqual(payload["sampler"], "er_sde")

    def test_cli_base_edit_and_generation_share_the_same_endpoint(self):
        for edit in (False, True):
            command = ["--model-variant", "base", "--prompt", "fox", "--output-resolution", "2048",
                       "--reference-megapixels", ".5"]
            if edit:
                command += ["--image", "ref.png"]
            args = self.client._build_parser().parse_args(command)
            with patch.object(self.client, "_submit_job", return_value="job") as submit, patch.object(
                self.client, "_poll_job", return_value={}
            ), patch.object(self.client, "_save_result", return_value=0), patch.object(self.client, "_image_to_data_url", return_value="base64"):
                method = self.client._run_edit if edit else self.client._run_generate
                self.assertEqual(method(args, endpoint_id="same-endpoint", api_key="key"), 0)
            self.assertEqual(submit.call_args.args[0], "same-endpoint")
            payload = submit.call_args.args[2]
            self.assertEqual(payload["model_options"], {"variant": "base", "output_resolution": 2048})
            self.assertEqual(payload["reference_megapixels"], .5)
            self.assertNotIn("model_variant", payload)
            self.assertNotIn("output_resolution", payload)
            self.assertEqual(payload["num_inference_steps"], 40)
            self.assertNotIn("seed", payload)

    def test_cli_krea_syntax(self):
        self.assertEqual(
            self.client._parse_lora("style@0.8, character.safetensors"),
            [
                {"name": "style", "strength": 0.8},
                {"name": "character", "strength": 1.0},
            ],
        )
        self.assertEqual(self.client._parse_lora(""), [])
        for text in ["style@nan", "style@2.1", "style@oops", "style,style"]:
            with self.subTest(text=text), self.assertRaises(SystemExit):
                self.client._parse_lora(text)

    def test_cli_submits_loras_for_both_modes(self):
        for edit in [False, True]:
            args_list = ["--prompt", "fox", "--lora", "style@0.8"]
            if edit:
                args_list += ["--image", "ref.png"]
            args = self.client._build_parser().parse_args(args_list)
            with (
                patch.object(self.client, "_submit_job", return_value="job") as submit,
                patch.object(self.client, "_poll_job", return_value={}),
                patch.object(self.client, "_save_result", return_value=0),
                patch.object(self.client, "_image_to_data_url", return_value="base64"),
            ):
                method = self.client._run_edit if edit else self.client._run_generate
                self.assertEqual(method(args, endpoint_id="endpoint", api_key="key"), 0)
            self.assertEqual(
                submit.call_args.args[2]["loras"], [{"name": "style", "strength": 0.8}]
            )


if __name__ == "__main__":
    unittest.main()
