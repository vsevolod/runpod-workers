"""Startup imports the handler without preselecting or downloading Turbo."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parents[2]))
from shared.tests.startup_contract import SHARED


class StartTests(unittest.TestCase):
    start = Path(__file__).parents[1] / "start.sh"

    def test_ready_handler_does_not_download_a_checkpoint_before_first_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(self.start, root / "start.sh")
            shutil.copytree(SHARED, root / "shared", ignore=shutil.ignore_patterns("tests", "__pycache__"))
            (root / "download_weights.py").write_text("raise RuntimeError('Do not preselect Turbo')")
            (root / "handler.py").write_text("""
import json, os
from pathlib import Path
Path(os.environ['OUTPUT']).write_text(json.dumps({key:os.environ.get(key) for key in ('LOCAL_FILES_ONLY','HF_HUB_CACHE')}))
""")
            result = subprocess.run(["sh", str(root / "start.sh")], env={
                **os.environ, "OUTPUT": str(root / "result.json"), "HF_HOME": str(root / "hf"),
                "LOCAL_FILES_ONLY": "0",
            }, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / "result.json").read_text()), {
                "LOCAL_FILES_ONLY": "1", "HF_HUB_CACHE": str(root / "hf/hub"),
            })
