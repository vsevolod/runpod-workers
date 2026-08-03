from __future__ import annotations

import io
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from download import (
    FilenameError,
    ItemNormalizeError,
    ItemResult,
    NormalizedItem,
    download_to_lora_dir,
    extract_filename_from_response,
    normalize_filename,
    normalize_item,
    resolve_api_url,
    resolve_lora_dir,
    run_batch,
)


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


class NormalizeItemTests(unittest.TestCase):
    def test_defaults(self):
        item = normalize_item({"model_version_id": "46846"})
        self.assertEqual(item, NormalizedItem("46846", True, None))

    def test_int_id(self):
        item = normalize_item({"model_version_id": 46846, "nsfw": False})
        self.assertEqual(item.model_version_id, "46846")
        self.assertIs(item.nsfw, False)

    def test_rejects_bool_id(self):
        with self.assertRaises(ItemNormalizeError):
            normalize_item({"model_version_id": True})

    def test_rejects_zero_and_bad_str(self):
        for raw in (0, -1, "", "12a", 1.5, None, "²"):
            with self.subTest(raw=raw):
                with self.assertRaises(ItemNormalizeError):
                    normalize_item({"model_version_id": raw})

    def test_rejects_string_nsfw(self):
        with self.assertRaises(ItemNormalizeError):
            normalize_item({"model_version_id": "1", "nsfw": "true"})

    def test_filename_override(self):
        item = normalize_item(
            {"model_version_id": "1", "filename": "x.safetensors"}
        )
        self.assertEqual(item.filename, "x.safetensors")

    def test_filename_null_is_error(self):
        # Key present with null is not "absent" — must be a string.
        with self.assertRaises(ItemNormalizeError):
            normalize_item({"model_version_id": "1", "filename": None})

    def test_bad_filename_override(self):
        with self.assertRaises(ItemNormalizeError):
            normalize_item({"model_version_id": "1", "filename": "x.zip"})

    def test_rejects_non_dict(self):
        with self.assertRaises(ItemNormalizeError):
            normalize_item("nope")  # type: ignore[arg-type]


class ResolveLoraDirTests(unittest.TestCase):
    def test_under_volume_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runpod-volume"
            root.mkdir()
            lora = root / "krea2" / "loras"
            with mock.patch("download.VOLUME_ROOT", root):
                got = resolve_lora_dir(env={"LORA_DIR": str(lora)})
            self.assertEqual(got, lora.resolve())
            self.assertTrue(got.is_dir())

    def test_rejects_outside_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runpod-volume"
            root.mkdir()
            outside = Path(tmp) / "other"
            outside.mkdir()
            with mock.patch("download.VOLUME_ROOT", root):
                with self.assertRaises(ValueError):
                    resolve_lora_dir(env={"LORA_DIR": str(outside)})

    def test_rejects_symlink_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runpod-volume"
            root.mkdir()
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            with mock.patch("download.VOLUME_ROOT", root):
                with self.assertRaises(ValueError):
                    resolve_lora_dir(env={"LORA_DIR": str(link)})


class FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes = b""):
        self.status = status
        self._headers = {k.lower(): v for k, v in headers.items()}
        self._buf = io.BytesIO(body)

    def getheader(self, name: str, default=None):
        return self._headers.get(name.lower(), default)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read() if n < 0 else self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class ScriptedOpener:
    """urlopen double: queue of FakeResponse or Exception."""

    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float = None):
        self.requests.append(req)
        if not self.script:
            raise AssertionError("unexpected urlopen call")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ResolveApiUrlTests(unittest.TestCase):
    def test_hosts(self):
        self.assertEqual(
            resolve_api_url("46846", True),
            "https://civitai.red/api/download/models/46846",
        )
        self.assertEqual(
            resolve_api_url("46846", False),
            "https://civitai.com/api/download/models/46846",
        )


class ExtractFilenameTests(unittest.TestCase):
    def test_content_disposition_header(self):
        resp = FakeResponse(
            200,
            {"Content-Disposition": 'attachment; filename="My LoRA.safetensors"'},
            b"x",
        )
        self.assertEqual(
            extract_filename_from_response(
                "https://cdn.example/file.bin", resp
            ),
            "My LoRA.safetensors",
        )

    def test_query_disposition(self):
        url = (
            "https://cdn.example/x?"
            "response-content-disposition=attachment%3B%20filename%3D%22q.safetensors%22"
        )
        resp = FakeResponse(200, {}, b"x")
        self.assertEqual(extract_filename_from_response(url, resp), "q.safetensors")

    def test_query_disposition_beats_header(self):
        # Spec precedence: query → header → path
        url = (
            "https://cdn.example/path/from_path.safetensors?"
            "response-content-disposition=attachment%3B%20filename%3D%22from_query.safetensors%22"
        )
        resp = FakeResponse(
            200,
            {"Content-Disposition": 'attachment; filename="from_header.safetensors"'},
            b"x",
        )
        self.assertEqual(
            extract_filename_from_response(url, resp),
            "from_query.safetensors",
        )

    def test_path_fallback(self):
        resp = FakeResponse(200, {}, b"x")
        self.assertEqual(
            extract_filename_from_response(
                "https://cdn.example/path/to/name.safetensors", resp
            ),
            "name.safetensors",
        )


class DownloadToLoraDirTests(unittest.TestCase):
    def _lora_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "runpod-volume"
        root.mkdir()
        lora = root / "krea2" / "loras"
        lora.mkdir(parents=True)
        return lora

    def test_bearer_only_on_first_request_then_write(self):
        lora = self._lora_dir()
        body = b"lora-bytes-here"
        opener = ScriptedOpener(
            [
                FakeResponse(
                    302,
                    {"Location": "https://cdn.example/dl/file.safetensors"},
                ),
                FakeResponse(
                    200,
                    {
                        "Content-Disposition": 'attachment; filename="file.safetensors"',
                        "Content-Length": str(len(body)),
                    },
                    body,
                ),
            ]
        )
        item = NormalizedItem("46846", True, None)
        result = download_to_lora_dir(
            item, lora, token="secret-token", urlopen=opener
        )
        self.assertEqual(result.status, "downloaded")
        self.assertEqual(result.filename, "file.safetensors")
        self.assertEqual(result.bytes, len(body))
        self.assertTrue((lora / "file.safetensors").is_file())
        self.assertEqual((lora / "file.safetensors").read_bytes(), body)
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(
            opener.requests[0].get_header("Authorization"),
            "Bearer secret-token",
        )
        # urllib stores headers with Header-Case; second hop must not send token
        self.assertIsNone(opener.requests[1].get_header("Authorization"))

    def test_rejects_http_redirect(self):
        lora = self._lora_dir()
        opener = ScriptedOpener(
            [FakeResponse(302, {"Location": "http://evil.example/x.safetensors"})]
        )
        result = download_to_lora_dir(
            NormalizedItem("1", True, None),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("HTTPS", result.error or "")

    def test_max_redirects(self):
        lora = self._lora_dir()
        # 6 HTTPS redirects → fail (max 5 hops means 6th redirect fails)
        script = [
            FakeResponse(302, {"Location": f"https://cdn.example/r{i}"})
            for i in range(6)
        ]
        opener = ScriptedOpener(script)
        result = download_to_lora_dir(
            NormalizedItem("1", True, None),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("redirect", (result.error or "").lower())

    def test_content_length_mismatch(self):
        lora = self._lora_dir()
        opener = ScriptedOpener(
            [
                FakeResponse(
                    200,
                    {
                        "Content-Disposition": 'attachment; filename="a.safetensors"',
                        "Content-Length": "100",
                    },
                    b"short",
                )
            ]
        )
        result = download_to_lora_dir(
            NormalizedItem("1", False, None),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertFalse((lora / "a.safetensors").exists())
        self.assertEqual(list(lora.glob("*.partial")), [])

    def test_skip_existing_with_override_without_http(self):
        lora = self._lora_dir()
        target = lora / "local.safetensors"
        target.write_bytes(b"already")
        opener = ScriptedOpener([])  # any call raises AssertionError
        result = download_to_lora_dir(
            NormalizedItem("99", True, "local.safetensors"),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "skipped")
        self.assertEqual(result.reason, "already_exists")
        self.assertEqual(opener.requests, [])

    def test_fail_non_regular_target_symlink(self):
        lora = self._lora_dir()
        real = lora / "real.safetensors"
        real.write_bytes(b"x")
        link = lora / "link.safetensors"
        link.symlink_to(real)
        opener = ScriptedOpener([])
        result = download_to_lora_dir(
            NormalizedItem("1", True, "link.safetensors"),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(opener.requests, [])

    def test_fail_directory_target(self):
        lora = self._lora_dir()
        (lora / "dir.safetensors").mkdir()
        opener = ScriptedOpener([])
        result = download_to_lora_dir(
            NormalizedItem("1", True, "dir.safetensors"),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(opener.requests, [])

    def test_network_error_is_failed_not_raise(self):
        lora = self._lora_dir()
        opener = ScriptedOpener(
            [urllib.error.URLError("dns failed")]
        )
        result = download_to_lora_dir(
            NormalizedItem("1", True, None),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.error)

    def test_http_404(self):
        lora = self._lora_dir()
        opener = ScriptedOpener([FakeResponse(404, {})])
        result = download_to_lora_dir(
            NormalizedItem("1", True, None),
            lora,
            token="t",
            urlopen=opener,
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("not found", (result.error or "").lower())


class RunBatchTests(unittest.TestCase):
    def _lora_dir(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "runpod-volume"
        root.mkdir()
        lora = root / "krea2" / "loras"
        lora.mkdir(parents=True)
        return lora

    def test_normalize_error_then_success(self):
        lora = self._lora_dir()
        body = b"ok-bytes"
        opener = ScriptedOpener(
            [
                FakeResponse(
                    200,
                    {
                        "Content-Disposition": 'attachment; filename="ok.safetensors"',
                        "Content-Length": str(len(body)),
                    },
                    body,
                )
            ]
        )
        out = run_batch(
            [
                {"model_version_id": True},  # invalid
                {"model_version_id": "2", "filename": "ok.safetensors"},
            ],
            token="t",
            lora_dir=lora,
            urlopen=opener,
        )
        self.assertEqual(out["summary"]["failed"], 1)
        self.assertEqual(out["summary"]["downloaded"], 1)
        self.assertEqual(out["summary"]["skipped"], 0)
        self.assertIn("note", out)
        self.assertEqual(out["dest"], str(lora))
        self.assertNotIn("output", out)

    def test_unicode_digit_id_fails_item_not_batch(self):
        """Unicode isdigit() chars must not abort the whole batch via ValueError."""
        lora = self._lora_dir()
        body = b"ok"
        opener = ScriptedOpener(
            [
                FakeResponse(
                    200,
                    {
                        "Content-Disposition": 'attachment; filename="ok.safetensors"',
                        "Content-Length": str(len(body)),
                    },
                    body,
                )
            ]
        )
        out = run_batch(
            [
                {"model_version_id": "²"},
                {"model_version_id": "2", "filename": "ok.safetensors"},
            ],
            token="t",
            lora_dir=lora,
            urlopen=opener,
        )
        self.assertEqual(out["summary"]["failed"], 1)
        self.assertEqual(out["summary"]["downloaded"], 1)
        self.assertEqual(out["results"][0]["status"], "failed")
        self.assertEqual(out["results"][1]["status"], "downloaded")
        self.assertTrue((lora / "ok.safetensors").is_file())

    def test_network_error_then_success(self):
        lora = self._lora_dir()
        body = b"second"
        opener = ScriptedOpener(
            [
                urllib.error.URLError("boom"),
                FakeResponse(
                    200,
                    {
                        "Content-Disposition": 'attachment; filename="b.safetensors"',
                        "Content-Length": str(len(body)),
                    },
                    body,
                ),
            ]
        )
        out = run_batch(
            [
                {"model_version_id": "1"},
                {"model_version_id": "2"},
            ],
            token="t",
            lora_dir=lora,
            urlopen=opener,
        )
        self.assertEqual(out["summary"]["failed"], 1)
        self.assertEqual(out["summary"]["downloaded"], 1)
        self.assertEqual(out["results"][0]["status"], "failed")
        self.assertEqual(out["results"][1]["status"], "downloaded")
        self.assertTrue((lora / "b.safetensors").is_file())

    def test_unexpected_exception_then_success(self):
        """Cover run_batch's bare except around download_to_lora_dir."""
        lora = self._lora_dir()
        calls = {"n": 0}

        def flaky_download(item, lora_dir, token, *, urlopen=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("unexpected boom")
            return ItemResult(
                model_version_id=item.model_version_id,
                status="downloaded",
                filename="ok.safetensors",
                path=str(lora_dir / "ok.safetensors"),
                bytes=3,
            )

        with mock.patch("download.download_to_lora_dir", side_effect=flaky_download):
            out = run_batch(
                [
                    {"model_version_id": "1"},
                    {"model_version_id": "2"},
                ],
                token="t",
                lora_dir=lora,
            )
        self.assertEqual(out["summary"]["failed"], 1)
        self.assertEqual(out["summary"]["downloaded"], 1)
        self.assertIn("unexpected boom", out["results"][0].get("error", ""))
        self.assertEqual(out["results"][1]["status"], "downloaded")


if __name__ == "__main__":
    unittest.main()
