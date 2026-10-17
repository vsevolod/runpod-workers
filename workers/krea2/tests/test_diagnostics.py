import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

import torch

spec = importlib.util.spec_from_file_location(
    "_krea2_diagnostics_tests", Path(__file__).parents[1] / "krea2_infer" / "diagnostics.py",
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class ProfilingTests(unittest.TestCase):
    def test_default_timings_do_not_synchronize_cuda(self):
        with (
            mock.patch.dict(os.environ, {"KREA_PROFILE_CUDA": "0"}),
            mock.patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            with diagnostics.log_stage("copy", name="adapter"):
                pass
        synchronize.assert_not_called()

    def test_profiling_synchronizes_around_work(self):
        events = []
        with (
            mock.patch.dict(os.environ, {"KREA_PROFILE_CUDA": "1"}),
            mock.patch.object(torch.cuda, "is_initialized", return_value=True),
            mock.patch.object(torch.cuda, "synchronize", side_effect=lambda: events.append("sync")),
        ):
            with diagnostics.log_stage("copy", name="adapter"):
                events.append("work")
        self.assertEqual(events, ["sync", "work", "sync"])

    def test_failure_preserves_original_exception(self):
        error = ValueError("bad weights")
        with self.assertRaises(ValueError) as caught:
            with diagnostics.log_stage("copy"):
                raise error
        self.assertIs(caught.exception, error)
