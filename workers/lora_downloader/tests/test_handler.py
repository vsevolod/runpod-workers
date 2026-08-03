from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from handler import handler


class HandlerTests(unittest.TestCase):
    def test_requires_input_key(self):
        out = handler({"items": [{"model_version_id": "1"}]})
        self.assertIn("error", out)

    def test_empty_items_error(self):
        out = handler({"input": {"items": []}})
        self.assertIn("error", out)

    def test_too_many_items_error(self):
        items = [{"model_version_id": str(i + 1)} for i in range(21)]
        out = handler({"input": {"items": items}})
        self.assertIn("error", out)

    def test_missing_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure CIVITAI_TOKEN absent; LORA_DIR may still matter after token check
            out = handler({"input": {"items": [{"model_version_id": "1"}]}})
        self.assertIn("error", out)
        self.assertIn("CIVITAI_TOKEN", out["error"])

    def test_success_shape_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "loras"
            lora.mkdir()
            env = {"CIVITAI_TOKEN": "tok", "LORA_DIR": str(lora)}
            fake_batch = {
                "dest": str(lora),
                "results": [
                    {
                        "model_version_id": "1",
                        "status": "downloaded",
                        "filename": "a.safetensors",
                    }
                ],
                "summary": {"downloaded": 1, "skipped": 0, "failed": 0},
                "note": "Restart warm krea2 workers to pick up new LoRA files.",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("handler.resolve_lora_dir", return_value=lora):
                    with mock.patch(
                        "handler.run_batch", return_value=fake_batch
                    ) as rb:
                        out = handler(
                            {
                                "input": {
                                    "items": [
                                        {
                                            "model_version_id": "1",
                                            "filename": "a.safetensors",
                                        }
                                    ]
                                }
                            }
                        )
            self.assertEqual(out, fake_batch)
            self.assertNotIn("output", out)
            rb.assert_called_once()
            self.assertEqual(rb.call_args.kwargs["token"], "tok")
            self.assertEqual(rb.call_args.kwargs["lora_dir"], lora)


if __name__ == "__main__":
    unittest.main()
