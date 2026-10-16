"""The container must finish downloading before it starts serving jobs."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


START = Path(__file__).parents[1] / "start.sh"


class StartTests(unittest.TestCase):
    def run_start(self, directory, download_exit=0):
        root = Path(directory)
        python = root / "python"
        # Stand in only for the external download/GPU processes. Run the real
        # entrypoint and check its sequencing and environment at both boundaries.
        python.write_text("#!/usr/bin/env python3\n" + '''
import json, os, sys
from pathlib import Path
is_download = Path(sys.argv[2]).name == "download_weights.py"
record = {key: os.environ.get(key) for key in
          ("MODEL_DIR", "HF_HOME", "LOCAL_FILES_ONLY", "LORA_DIR", "HF_TOKEN", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")}
record["process"] = "download" if is_download else "handler"
with open(os.environ["START_RECORD"], "a") as output:
    output.write(json.dumps(record) + "\\n")
if is_download:
    if int(os.environ["DOWNLOAD_EXIT"]):
        sys.exit(int(os.environ["DOWNLOAD_EXIT"]))
    Path(os.environ["DOWNLOAD_READY"]).touch()
elif not Path(os.environ["DOWNLOAD_READY"]).exists():
    sys.exit(99)
''')
        python.chmod(0o755)
        env = {
            **os.environ,
            "PATH": str(root) + os.pathsep + os.environ["PATH"],
            "MODEL_DIR": str(root / "models"),
            "HF_HOME": str(root / "hf"),
            "HF_HUB_CACHE": "/old/remote/hub",
            "HUGGINGFACE_HUB_CACHE": "/old/remote/hub",
            "LORA_DIR": "/runpod-volume/qwen_image_21_turbo/loras",
            "HF_TOKEN": "test-token",
            "LOCAL_FILES_ONLY": "0",
            "START_RECORD": str(root / "calls.jsonl"),
            "DOWNLOAD_READY": str(root / "ready"),
            "DOWNLOAD_EXIT": str(download_exit),
        }
        result = subprocess.run(["sh", str(START)], env=env, capture_output=True, text=True)
        records = root / "calls.jsonl"
        calls = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
        return result, calls

    def test_handler_starts_only_after_download_and_uses_the_same_local_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_start(tmp)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([call["process"] for call in calls], ["download", "handler"])
            self.assertEqual(calls[1]["MODEL_DIR"], str(Path(tmp) / "models"))
            self.assertEqual(calls[1]["HF_HOME"], str(Path(tmp) / "hf"))
            for call in calls:
                self.assertEqual(call["HF_HUB_CACHE"], str(Path(tmp) / "hf/hub"))
                self.assertEqual(call["HUGGINGFACE_HUB_CACHE"], str(Path(tmp) / "hf/hub"))
            self.assertEqual(calls[1]["LOCAL_FILES_ONLY"], "1")
            self.assertEqual(calls[1]["LORA_DIR"], "/runpod-volume/qwen_image_21_turbo/loras")
            self.assertEqual(calls[0]["HF_TOKEN"], "test-token")

    def test_download_failure_exits_without_starting_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self.run_start(tmp, download_exit=7)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual([call["process"] for call in calls], ["download"])
