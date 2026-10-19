"""Starting the common worker must not pick or download a checkpoint."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class StartupTests(unittest.TestCase):
    def test_bootstrap_reaches_handler_without_starting_any_downloader(self):
        worker = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp)
            shutil.copy(worker / 'start.sh', app / 'start.sh')
            shutil.copytree(worker.parent / 'shared', app / 'shared',
                            ignore=shutil.ignore_patterns('tests', '__pycache__'))
            (app / 'download_weights.py').write_text("raise RuntimeError('No model was selected')")
            (app / 'handler.py').write_text("from pathlib import Path\nPath(__file__).with_name('ready').touch()\n")
            result = subprocess.run(['sh', str(app / 'start.sh')],
                                    env={**os.environ, 'HF_HOME': str(app / 'hf')},
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((app / 'ready').is_file())
