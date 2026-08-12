"""image_input: base64 / data URL / size guards / staging."""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from image_input import (  # noqa: E402
    ImageInputError,
    cleanup_staged,
    resolve_image_ref,
    stage_image,
)


def _png_bytes(w: int = 8, h: int = 8, color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


class TestResolveImageRef(unittest.TestCase):
    def test_raw_base64_png(self):
        raw = _png_bytes()
        out, ext = resolve_image_ref(_b64(raw))
        self.assertEqual(ext, "png")
        self.assertTrue(out.startswith(b"\x89PNG"))

    def test_data_url(self):
        raw = _png_bytes()
        ref = f"data:image/png;base64,{_b64(raw)}"
        out, ext = resolve_image_ref(ref)
        self.assertEqual(ext, "png")
        self.assertTrue(out.startswith(b"\x89PNG"))

    def test_empty_raises(self):
        with self.assertRaises(ImageInputError):
            resolve_image_ref("")

    def test_bad_scheme_raises(self):
        with self.assertRaises(ImageInputError) as ctx:
            resolve_image_ref("file:///etc/passwd")
        self.assertIn("scheme", str(ctx.exception).lower())

    def test_oversized_bytes(self):
        raw = _png_bytes()
        with self.assertRaises(ImageInputError):
            resolve_image_ref(_b64(raw), max_bytes=10)

    def test_max_pixels(self):
        raw = _png_bytes(64, 64)
        with self.assertRaises(ImageInputError):
            resolve_image_ref(_b64(raw), max_pixels=100)

    def test_fetch_url(self):
        raw = _png_bytes()

        class _Resp:
            headers: dict = {}

            def read(self, n: int = -1):
                if getattr(self, "_done", False):
                    return b""
                self._done = True
                return raw

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with mock.patch("image_input.urlopen", return_value=_Resp()):
            out, ext = resolve_image_ref("https://example.com/a.png")
        self.assertEqual(ext, "png")
        self.assertTrue(out.startswith(b"\x89PNG"))


class TestStageImage(unittest.TestCase):
    def test_writes_basename(self):
        raw = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            name = stage_image(
                _b64(raw),
                input_dir=Path(td),
                basename_stem="job1_first",
            )
            self.assertEqual(name, "job1_first.png")
            self.assertTrue((Path(td) / name).is_file())
            cleanup_staged(Path(td), name)
            self.assertFalse((Path(td) / name).exists())


if __name__ == "__main__":
    unittest.main()
