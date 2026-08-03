from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "download_lora.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("download_lora_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class CliHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli()

    def test_exit_codes(self):
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"summary": {"failed": 0}}}
            ),
            0,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"summary": {"failed": 1}}}
            ),
            1,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result({"status": "FAILED"}),
            1,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"error": "missing token"}}
            ),
            1,
        )

    def test_build_job_input_sfw(self):
        inp = self.cli.build_job_input(
            ["1", "2"], filenames={"1": "a.safetensors"}, sfw=True
        )
        self.assertEqual(
            inp,
            {
                "items": [
                    {
                        "model_version_id": "1",
                        "nsfw": False,
                        "filename": "a.safetensors",
                    },
                    {"model_version_id": "2", "nsfw": False},
                ]
            },
        )

    def test_build_job_input_nsfw_default(self):
        inp = self.cli.build_job_input(["9"], filenames={}, sfw=False)
        self.assertTrue(inp["items"][0]["nsfw"])


if __name__ == "__main__":
    unittest.main()
