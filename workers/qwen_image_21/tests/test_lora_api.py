import base64
import importlib.util
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from qwen_image_21_infer.lora import LoRACatalog, TURBO_ADAPTER_NAME
from qwen_image_21_infer.pipeline import QwenImage21Runtime, _load_turbo_lora
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
        self.calls = []
        self.failure = None

    def __call__(self, **kwargs):
        self.calls.append((kwargs, list(self.active)))
        if self.failure:
            raise self.failure
        return SimpleNamespace(images=[Image.new("RGB", (32, 32))])


class LoRAAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "style.safetensors").write_bytes(b"weights")
        self.pipe = InferencePipeline()
        runtime = QwenImage21Runtime(self.pipe, LoRACatalog.scan(self.tmp.name))
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
        with patch("qwen_image_21_infer.pipeline.load_pipeline", return_value=runtime):
            self.handler = load_module("_qwen_lora_handler", WORKER / "handler.py")
        self.handler._save_and_upload_images = lambda images, job_id: [
            "data:image/png;base64,test"
        ]

    def test_generation_and_edit_forward_loras_and_report_effective_selection(self):
        buffer = BytesIO()
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
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
            self.assertEqual(adapters[0], (TURBO_ADAPTER_NAME, 1.0))
            self.assertEqual(adapters[1][1], 0.8)
            self.assertEqual(self.pipe.loaded, {TURBO_ADAPTER_NAME})
        self.assertIn("image", self.pipe.calls[-1][0])

    def test_omitted_loras_preserve_existing_api(self):
        output = self.handler.generate_image({"input": {"prompt": "fox"}})
        self.assertEqual(output["loras"], [])
        self.assertEqual(self.pipe.calls[-1][1], [(TURBO_ADAPTER_NAME, 1.0)])

    def test_invalid_name_and_incompatible_weights_do_not_refresh_worker(self):
        for name in ["missing", "style"]:
            self.pipe.state = {}
            output = self.handler.generate_image(
                {"input": {"prompt": "fox", "loras": [{"name": name}]}}
            )
            self.assertIn("error", output)
            self.assertNotIn("refresh_worker", output)
            self.assertEqual(self.pipe.calls, [])
        self.assertEqual(self.pipe.loaded, {TURBO_ADAPTER_NAME})

    def test_oom_keeps_restart_behavior_and_restores_turbo(self):
        self.pipe.failure = OutOfMemoryError("OOM")
        output = self.handler.generate_image(
            {"input": {"prompt": "fox", "loras": [{"name": "style"}]}}
        )
        self.assertTrue(output["refresh_worker"])
        self.assertEqual(self.pipe.loaded, {TURBO_ADAPTER_NAME})

    def test_builtin_adapter_has_stable_name_and_unit_strength(self):
        pipe = Mock()
        with patch(
            "qwen_image_21_infer.pipeline.resolve_lora_source",
            return_value=("/volume", "turbo.safetensors", True),
        ):
            _load_turbo_lora(pipe, None, local_files_only=True)
        self.assertEqual(
            pipe.load_lora_weights.call_args.kwargs["adapter_name"], TURBO_ADAPTER_NAME
        )
        pipe.set_adapters.assert_called_once_with(
            [TURBO_ADAPTER_NAME], adapter_weights=[1.0]
        )


class ClientLoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = load_module(
            "_qwen_lora_client", WORKER.parents[1] / "scripts" / "qwen_image_21.py"
        )

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
