"""Active model ownership must end before its replacement starts loading."""

import gc
from pathlib import Path
import sys
import threading
import unittest
import weakref

sys.path.insert(0, str(Path(__file__).parents[1]))


class ModelManagerTests(unittest.TestCase):
    def manager(self, registry):
        try:
            from image_worker.manager import ModelManager
        except ImportError:
            self.fail('The common model ownership manager is missing')
        return ModelManager(registry)

    def test_lazy_initialization_warm_reuse_and_release_before_replacement_load(self):
        events, references = [], []

        def factory(name):
            class Backend:
                def load(self):
                    if references:
                        gc.collect()
                        self_ref = references[-1]
                        self_test.assertIsNone(self_ref(), 'Previous backend is still retained')
                    references.append(weakref.ref(self))
                    events.append(('load', name))

                def infer(self, norm, seed, raw_loras):
                    events.append(('infer', name, seed, raw_loras))
                    return name

                def close(self):
                    events.append(('close', name))
            return Backend

        self_test = self
        manager = self.manager({name: factory(name) for name in ('krea2', 'qwen_image_21')})
        self.assertEqual(events, [])
        for model in ('krea2', 'krea2', 'qwen_image_21', 'krea2'):
            self.assertEqual(manager.infer(model, object(), 0, []), model)
        manager.close()
        manager.close()
        self.assertEqual([e for e in events if e[0] in ('load', 'close')], [
            ('load', 'krea2'), ('close', 'krea2'), ('load', 'qwen_image_21'),
            ('close', 'qwen_image_21'), ('load', 'krea2'), ('close', 'krea2'),
        ])
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

    def test_failed_load_is_closed_and_next_request_can_retry(self):
        events = []

        class Backend:
            def load(self):
                events.append('load')
                if events.count('load') == 1:
                    raise RuntimeError('interrupted checkpoint loading')

            def close(self):
                events.append('close')

            def infer(self, norm, seed, raw_loras):
                return seed

        manager = self.manager({'krea2': Backend})
        with self.assertRaisesRegex(RuntimeError, 'interrupted checkpoint'):
            manager.infer('krea2', None, 7, [])
        self.assertEqual(events, ['load', 'close'])
        self.assertEqual(manager.infer('krea2', None, 0, []), 0)
        self.assertEqual(events, ['load', 'close', 'load'])

    def test_unknown_model_does_not_close_the_resident_backend(self):
        events = []

        class Backend:
            def load(self):
                events.append('load')

            def close(self):
                events.append('close')

            def infer(self, *args):
                return 'warm'

        manager = self.manager({'krea2': Backend})
        manager.infer('krea2', None, 1, [])
        with self.assertRaises(ValueError):
            manager.infer('unknown', None, 1, [])
        self.assertEqual(manager.infer('krea2', None, 1, []), 'warm')
        self.assertEqual(events, ['load'])

    def test_switch_waits_until_current_inference_returns(self):
        entered, finish, second_loaded = threading.Event(), threading.Event(), threading.Event()
        errors = []

        class First:
            def load(self):
                pass

            def close(self):
                pass

            def infer(self, *args):
                entered.set()
                if not finish.wait(3):
                    raise TimeoutError('test inference was not released')

        class Second(First):
            def load(self):
                second_loaded.set()

            def infer(self, *args):
                return 'second'

        manager = self.manager({'first': First, 'second': Second})

        def run(name):
            try:
                manager.infer(name, None, 0, [])
            except BaseException as error:
                errors.append(error)

        first = threading.Thread(target=run, args=('first',))
        second = threading.Thread(target=run, args=('second',))
        first.start()
        try:
            self.assertTrue(entered.wait(2))
            second.start()
            self.assertFalse(second_loaded.wait(.1))
        finally:
            finish.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_loaded.is_set())
