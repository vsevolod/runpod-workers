import importlib
import os
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

WORKER = Path(__file__).parents[1]
sys.path.insert(0, str(WORKER))


class DiagnosticsTests(unittest.TestCase):
    def diagnostics(self):
        try:
            return importlib.import_module("qwen_image_21_infer.diagnostics")
        except ModuleNotFoundError:
            self.fail("Startup diagnostics are not implemented")

    def test_stage_failure_logs_traceback_and_preserves_exception(self):
        diagnostics = self.diagnostics()
        failure = RuntimeError("cannot place weights")
        with self.assertLogs(diagnostics.logger, level="INFO") as logs:
            with self.assertRaises(RuntimeError) as caught:
                with diagnostics.log_stage("cuda_placement"):
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("stage=cuda_placement begin", logs.output[0])
        self.assertIn("stage=cuda_placement failed", logs.output[-1])
        self.assertIn("Traceback", logs.output[-1])
        self.assertNotIn(" done ", "\n".join(logs.output))

    def test_missing_memory_metrics_do_not_interrupt_stage(self):
        diagnostics = self.diagnostics()
        with patch.object(Path, "read_text", side_effect=OSError("unavailable")):
            with self.assertLogs(diagnostics.logger, level="INFO") as logs:
                with diagnostics.log_stage("weights"):
                    pass
        self.assertIn("stage=weights done elapsed_s=", logs.output[-1])

    def test_stack_watchdog_repeats_only_during_startup_and_never_exits(self):
        self.diagnostics()
        for fail in (False, True):
            with self.subTest(fail=fail):
                script = f"""
import time
from qwen_image_21_infer.diagnostics import startup_diagnostics
try:
    with startup_diagnostics():
        time.sleep(0.16)
        if {fail!r}:
            raise RuntimeError('load failed')
except RuntimeError:
    pass
print('WATCHDOG_CANCELLED', flush=True, file=__import__('sys').stderr)
time.sleep(0.16)
"""
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    env={
                        **os.environ,
                        "PYTHONPATH": str(WORKER),
                        "QWEN_STARTUP_TRACE_INTERVAL": "0.04",
                    },
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                before, after = result.stderr.split("WATCHDOG_CANCELLED", 1)
                self.assertGreaterEqual(before.count("Timeout"), 2)
                self.assertIn("<module>", before)
                self.assertNotIn("Timeout", after)

    def test_zero_interval_disables_stack_dumping(self):
        diagnostics = self.diagnostics()
        with patch.dict(os.environ, {"QWEN_STARTUP_TRACE_INTERVAL": "0"}):
            with patch.object(
                diagnostics.faulthandler,
                "dump_traceback",
                side_effect=AssertionError("watchdog must stay disabled"),
            ):
                with diagnostics.startup_diagnostics():
                    pass

    def test_invalid_interval_falls_back_without_interrupting_startup(self):
        diagnostics = self.diagnostics()
        for value in ("invalid", "nan", "inf", "-1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"QWEN_STARTUP_TRACE_INTERVAL": value}):
                    with self.assertLogs(diagnostics.logger, level="INFO") as logs:
                        with diagnostics.startup_diagnostics():
                            pass
                self.assertIn("using 60 seconds", logs.output[0])
                self.assertIn("stage=startup done", logs.output[-1])

    def test_unavailable_stderr_descriptor_does_not_interrupt_startup(self):
        diagnostics = self.diagnostics()
        for stderr in (StringIO(), None):
            with self.subTest(stderr=stderr):
                with patch.object(sys, "stderr", stderr):
                    with self.assertLogs(diagnostics.logger, level="INFO") as logs:
                        with diagnostics.startup_diagnostics():
                            pass
                self.assertIn("Could not enable startup stack dumps", logs.output[0])
                self.assertIn("stage=startup done", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
