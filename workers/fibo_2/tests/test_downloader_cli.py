"""The standalone downloader exposes its shared cache/output arguments."""

from pathlib import Path
import subprocess
import sys
import unittest


class DownloaderCLITests(unittest.TestCase):
    def test_help_includes_common_output_and_cache_options(self):
        script = Path(__file__).parents[1] / 'download_weights.py'
        result = subprocess.run([sys.executable, str(script), '--help'],
                                capture_output=True, text=True, check=True)
        for option in ('--output', '--hf-home', '--token', '--variant'):
            self.assertIn(option, result.stdout)


if __name__ == '__main__':
    unittest.main()
