"""One selected pipeline per worker, with serialized switching and warm-file repair."""

import gc
from contextlib import contextmanager
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

sys.path.insert(0, str(Path(__file__).parents[1]))


class ModelManagerTests(unittest.TestCase):
    def manager_class(self):
        try:
            from qwen_image_21_infer.model_manager import ModelManager
        except ImportError:
            self.fail("Request-selected model manager is not implemented")
        return ModelManager

    def test_lazy_first_base_reuses_warm_model_and_releases_before_loading_turbo(self):
        events, references = [], []
        shared_vae = object()

        class Runtime:
            def __init__(self, variant):
                self.pipe = SimpleNamespace(vae=shared_vae)
                self.variant = variant

            def run(self, norm, seed, loras):
                events.append(("infer", self.variant, seed))
                return [self.variant]

            def close(self):
                events.append(("close", self.variant))
                self.pipe = None

        @contextmanager
        def prepare(variant):
            events.append(("check_files", variant))
            yield None

        def load(*, model_variant, vae=None):
            if references:
                gc.collect()
                self.assertIsNone(references[-1](), "The old runtime must be released before loading its replacement")
                self.assertIs(vae, shared_vae)
            events.append(("load", model_variant))
            runtime = Runtime(model_variant)
            references.append(weakref.ref(runtime))
            return runtime

        manager = self.manager_class()(loader=load, prepare=prepare)
        self.assertEqual(events, [], "Initialization must not download or load Turbo")
        for variant in ["base", "base", "turbo", "base"]:
            result = manager.run(SimpleNamespace(model_variant=variant), seed=7, loras=())
            self.assertEqual(result, [variant])
        self.assertEqual([e for e in events if e[0] == "load"], [("load", "base"), ("load", "turbo"), ("load", "base")])
        self.assertEqual([e for e in events if e[0] == "check_files"], [("check_files", v) for v in ["base", "base", "turbo", "base"]])

    def test_missing_files_on_a_warm_model_are_repaired_before_inference(self):
        events = []
        runtime = SimpleNamespace(pipe=SimpleNamespace(vae=object()), close=Mock(), run=lambda *a, **k: events.append("infer"))
        pending = SimpleNamespace(wait=lambda component: events.append("repaired"))

        @contextmanager
        def prepare(variant):
            yield pending

        manager = self.manager_class()(loader=lambda **kwargs: runtime, prepare=prepare)
        manager.run(SimpleNamespace(model_variant="turbo"), seed=1, loras=())
        manager.run(SimpleNamespace(model_variant="turbo"), seed=2, loras=())
        self.assertEqual(events, ["repaired", "infer", "repaired", "infer"])

    def test_failed_switch_leaves_no_stale_runtime_and_can_be_retried(self):
        old = SimpleNamespace(pipe=SimpleNamespace(vae=object()), run=Mock(), close=Mock())
        replacement = SimpleNamespace(pipe=SimpleNamespace(vae=object()), run=Mock(return_value=["base"]))
        loader = Mock(side_effect=[old, RuntimeError("interrupted load"), replacement])

        @contextmanager
        def prepare(variant):
            yield None

        manager = self.manager_class()(loader=loader, prepare=prepare)
        manager.run(SimpleNamespace(model_variant="turbo"), seed=1, loras=())
        with self.assertRaisesRegex(RuntimeError, "interrupted load"):
            manager.run(SimpleNamespace(model_variant="base"), seed=1, loras=())
        old.close.assert_called_once()
        self.assertEqual(manager.run(SimpleNamespace(model_variant="base"), seed=1, loras=()), ["base"])
        self.assertIsNone(loader.call_args.kwargs["vae"], "A failed load must not leave shared modules with partial offload hooks")

    def test_downloading_base_keeps_an_already_valid_shared_vae(self):
        vae = object()
        loads = []
        def load(**kwargs):
            loads.append(kwargs)
            return SimpleNamespace(pipe=SimpleNamespace(vae=vae), run=lambda *a, **kw: [], close=Mock())
        @contextmanager
        def prepare(variant):
            yield SimpleNamespace(reload_vae=False, wait=Mock()) if variant == "base" else None
        manager = self.manager_class()(loader=load, prepare=prepare)
        manager.run(SimpleNamespace(model_variant="turbo"), seed=1, loras=())
        manager.run(SimpleNamespace(model_variant="base"), seed=1, loras=())
        self.assertIsNone(loads[0]["vae"])
        self.assertIs(loads[1]["vae"], vae)

    def test_switch_waits_for_an_in_flight_generation(self):
        entered, release, second_prepare = threading.Event(), threading.Event(), threading.Event()
        errors = []

        class Runtime:
            pipe = SimpleNamespace(vae=object())
            def run(self, norm, **kwargs):
                if norm.model_variant == "turbo":
                    entered.set()
                    if not release.wait(5):
                        raise TimeoutError("test generation was not released")
                return [norm.model_variant]
            def close(self):
                pass

        @contextmanager
        def prepare(variant):
            if variant == "base":
                second_prepare.set()
            yield None

        manager = self.manager_class()(loader=lambda **kwargs: Runtime(), prepare=prepare)
        def run(variant):
            try:
                manager.run(SimpleNamespace(model_variant=variant), seed=1, loras=())
            except BaseException as error:
                errors.append(error)
        first = threading.Thread(target=run, args=("turbo",))
        second = threading.Thread(target=run, args=("base",))
        first.start()
        try:
            self.assertTrue(entered.wait(3))
            second.start()
            self.assertFalse(second_prepare.wait(.1))
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertEqual(errors, [])
        self.assertTrue(second_prepare.is_set())


if __name__ == "__main__":
    unittest.main()
