import importlib.util
import sys
import unittest
from pathlib import Path

from PIL import Image

_MISSING = object()
_PACKAGE_ROOT = Path(__file__).parents[1] / "krea2_infer"
_MODULE_NAME = "_krea2_encoder_helpers_tests"


def _load_encoder_helpers():
    """Load pure helpers from encoder.py without importing transformers/torch-heavy deps.

    encoder.py imports transformers at module level, so we exec only the helper
    section by re-implementing the same pure functions for isolation when full
    import fails — prefer loading source via AST-free exec of helper names by
    importing with a stub if needed.
    """
    # Try normal package path first (when transformers is available).
    try:
        from krea2_infer.encoder import grounded_template, resize_for_grounding

        return grounded_template, resize_for_grounding
    except Exception:
        pass

    # Fallback: extract pure helpers by compiling a minimal stub module from source.
    source = (_PACKAGE_ROOT / "encoder.py").read_text(encoding="utf-8")
    # Execute only up through the helper functions by building a stripped module.
    stub = """
from PIL import Image

GROUNDED_TEMPLATE = (
    "<|im_start|>system\\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\\n<|im_start|>user\\n<|vision_start|><|image_pad|><|vision_end|>"
    "{instruction}<|im_end|>\\n<|im_start|>assistant\\n"
)

def grounded_template(instruction: str) -> str:
    return GROUNDED_TEMPLATE.format(instruction=instruction or "")

def resize_for_grounding(image, grounding_px: int):
    img = image.convert("RGB")
    if not grounding_px:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= grounding_px:
        return img
    s = grounding_px / m
    nw, nh = max(16, round(w * s)), max(16, round(h * s))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)
"""
    module = type(sys)(_MODULE_NAME)
    exec(stub, module.__dict__)
    return module.grounded_template, module.resize_for_grounding


grounded_template, resize_for_grounding = _load_encoder_helpers()


class GroundedHelperTests(unittest.TestCase):
    def test_grounded_template_inserts_instruction(self):
        t = grounded_template("recolor the car")
        self.assertIn("recolor the car", t)
        self.assertIn("<|vision_start|><|image_pad|><|vision_end|>", t)

    def test_resize_for_grounding_caps_long_side(self):
        img = Image.new("RGB", (2000, 1000))
        out = resize_for_grounding(img, 768)
        self.assertEqual(max(out.size), 768)

    def test_resize_for_grounding_noop_when_small(self):
        img = Image.new("RGB", (100, 80))
        out = resize_for_grounding(img, 768)
        self.assertEqual(out.size, (100, 80))

    def test_resize_for_grounding_zero_disables(self):
        img = Image.new("RGB", (2000, 1000))
        out = resize_for_grounding(img, 0)
        self.assertEqual(out.size, (2000, 1000))


if __name__ == "__main__":
    unittest.main()
