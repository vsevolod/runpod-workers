"""The standalone Krea client emits the canonical shared API shape."""

import importlib.util
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch


class KreaAPIClientTests(unittest.TestCase):
    def test_v2_edit_presets_select_the_correct_lora_and_reference_defaults(self):
        path = Path(__file__).parents[3] / "scripts/krea2_image.py"
        spec = importlib.util.spec_from_file_location("_krea_v2_client", path)
        client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "ref.png"
            from PIL import Image
            Image.new("RGB", (64, 96)).save(image)
            for mode, boost, position in (("identity", 2., "beside"), ("general", 4., "aligned")):
                with self.subTest(mode=mode):
                    args = client._build_parser().parse_args([
                        "--image", str(image), "--prompt", "edit", "--edit-model", mode,
                    ])
                    with patch.object(client, "_submit_job", return_value="job") as submit, patch.object(
                        client, "_poll_job", return_value={}
                    ), patch.object(client, "_save_result", return_value=0):
                        self.assertEqual(client._run_edit(args, endpoint_id="endpoint", api_key="key"), 0)
                    payload = submit.call_args.args[2]
                    self.assertEqual(payload["reference_megapixels"], "max")
                    self.assertNotIn("reference_megapixels", payload["model_options"])
                    self.assertEqual(payload["loras"], [{"name": f"krea2_{mode}_edit_v2", "strength": 1.}])
                    self.assertEqual(payload["model_options"]["ref_boost"], boost)
                    self.assertEqual(payload["model_options"]["ref_boost_a"], 1.)
                    self.assertEqual(payload["model_options"]["ref_position"], position)
                    self.assertEqual(payload["model_options"]["ref_position_a"], "aligned")

    def test_generation_and_edit_send_nested_options_and_explicit_euler(self):
        path = Path(__file__).parents[3] / "scripts/krea2_image.py"
        spec = importlib.util.spec_from_file_location("_krea_api_client", path)
        client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        image_path = Path(tmp.name) / "ref.png"
        from PIL import Image
        Image.new("RGB", (64, 96)).save(image_path)
        for edit in (False, True):
            command = ["--prompt", "fox", "--lora", "", "--mu", ".9", "--seed", "0"]
            if edit:
                command += ["--image", str(image_path), "--ref-boost", "4", "--grounding-px", "0", "--fit-mode", "crop"]
            args = client._build_parser().parse_args(command)
            with patch.object(client, "_submit_job", return_value="job") as submit, patch.object(
                client, "_poll_job", return_value={}
            ), patch.object(client, "_save_result", return_value=0):
                method = client._run_edit if edit else client._run_generate
                self.assertEqual(method(args, endpoint_id="endpoint", api_key="key"), 0)
            payload = submit.call_args.args[2]
            self.assertEqual(payload["reference_megapixels"], "max")
            self.assertEqual(payload["model_options"]["mu"], .9)
            self.assertEqual(payload["sampler"], "euler")
            self.assertEqual(payload["seed"], 0)
            for name in ("mu", "ref_boost", "grounding_px", "fit_mode"):
                self.assertNotIn(name, payload)
            if edit:
                self.assertEqual(payload["model_options"], {"mu": .9, "ref_boost": 4., "grounding_px": 0, "fit_mode": "crop",
                                                           "ref_boost_a": 1., "ref_position": "beside", "ref_position_a": "aligned"})


if __name__ == "__main__":
    unittest.main()
