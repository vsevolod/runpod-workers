import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image

_MISSING = object()
_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_MODULE_NAME = "_krea2_edit_geometry_tests"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        _PACKAGE_ROOT / "edit_geometry.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load edit_geometry module")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(_MODULE_NAME, _MISSING)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is _MISSING:
            sys.modules.pop(_MODULE_NAME, None)
        else:
            sys.modules[_MODULE_NAME] = previous
    return module


_geo = _load_module()
target_size_from_source = _geo.target_size_from_source
fit_source_pixels = _geo.fit_source_pixels


class TargetSizeTests(unittest.TestCase):
    def test_snaps_to_16_and_caps_mp(self):
        # 2000x2000 = 4MP → scale to max_megapixels=1.0 → ~1000^2, snap down to *16
        w, h = target_size_from_source(2000, 2000, max_megapixels=1.0, align=16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertLessEqual(w * h, 1_000_000 + 16 * 16)

    def test_preserves_ar_approximately(self):
        w, h = target_size_from_source(1920, 1080, max_megapixels=1.5, align=16)
        # /16 snap can shift AR slightly; require within ~1% relative error.
        self.assertAlmostEqual(w / h, 1920 / 1080, delta=0.02)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)


class FitSourcePixelsTests(unittest.TestCase):
    def test_crop_returns_exact_target_hw(self):
        img = Image.new("RGB", (300, 200), (10, 20, 30))
        out = fit_source_pixels(img, target_h=256, target_w=512, fit_mode="crop")
        self.assertEqual(out.size, (512, 256))  # PIL (W,H)

    def test_fit_matched_ar_fills_target(self):
        img = Image.new("RGB", (512, 256), (1, 2, 3))
        out = fit_source_pixels(img, target_h=256, target_w=512, fit_mode="fit")
        self.assertEqual(out.size, (512, 256))

    def test_fit_mismatched_ar_may_be_smaller_or_equal(self):
        # portrait source, landscape target
        img = Image.new("RGB", (200, 400), (0, 0, 0))
        out = fit_source_pixels(img, target_h=256, target_w=512, fit_mode="fit")
        ow, oh = out.size
        self.assertLessEqual(ow, 512)
        self.assertLessEqual(oh, 256)
        self.assertEqual(ow % 16, 0)
        self.assertEqual(oh % 16, 0)
        self.assertGreater(ow * oh, 0)


if __name__ == "__main__":
    unittest.main()
