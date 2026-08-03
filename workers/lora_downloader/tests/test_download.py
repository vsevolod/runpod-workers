from __future__ import annotations

import unittest

from download import FilenameError, normalize_filename


class NormalizeFilenameTests(unittest.TestCase):
    def test_accepts_simple(self):
        self.assertEqual(normalize_filename("style.safetensors"), "style.safetensors")

    def test_accepts_unicode_and_spaces(self):
        name = "My LoRA 日本語.safetensors"
        self.assertEqual(normalize_filename(name), name)

    def test_strips_ends_keeps_inner_spaces(self):
        self.assertEqual(
            normalize_filename("  a b.safetensors  "),
            "a b.safetensors",
        )

    def test_rejects_wrong_suffix(self):
        for bad in (
            "a.zip",
            "a.ckpt",
            "a.SafeTensors",
            "a.safetensor",
            "a.safetensors.bak",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(FilenameError):
                    normalize_filename(bad)

    def test_rejects_empty_stem(self):
        with self.assertRaises(FilenameError):
            normalize_filename(".safetensors")

    def test_rejects_dot_stems(self):
        # "..safetensors" → stem "."; "...safetensors" → stem ".."
        with self.assertRaises(FilenameError):
            normalize_filename("..safetensors")
        with self.assertRaises(FilenameError):
            normalize_filename("...safetensors")

    def test_rejects_path_chars_and_controls(self):
        for bad in (
            "a/b.safetensors",
            "a\\b.safetensors",
            "a\nb.safetensors",
            "a\x00b.safetensors",
        ):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(FilenameError):
                    normalize_filename(bad)

    def test_rejects_non_str(self):
        with self.assertRaises(FilenameError):
            normalize_filename(123)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
