# lora_downloader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin RunPod Serverless CPU worker that batch-downloads CivitAI LoRA `.safetensors` files into `$LORA_DIR` on the shared network volume for krea2.

**Architecture:** Original stdlib `urllib` download (no GPL vendor). Job-level validation for `items` list bounds via `rp_validator` constraints + `LORA_DIR` under hard-coded `/runpod-volume`; per-item normalize in a loop with partial success (including network/OS exceptions); sequential downloads; atomic tempfile write + rename; skip-if-exists **before** HTTP when filename override is set; deploy invariant `workersMax=1`. Flat handler return. CLI requires `ENDPOINT_ID`, supports `--sfw`, unwraps RunPod envelope, non-zero exit on job error or `summary.failed > 0`.

**Tech Stack:** Python 3.12, `runpod` SDK, stdlib `urllib`/`tempfile`/`unittest`, Docker `python:3.12-slim`

**Spec:** `docs/superpowers/specs/2026-08-03-lora-downloader-design.md` (rev 3)

---

## File map

| File | Responsibility |
|------|----------------|
| `workers/lora_downloader/download.py` | Filename/item normalize, LORA_DIR resolve, CivitAI HTTP, atomic write, `run_batch` |
| `workers/lora_downloader/schemas.py` | Job-level `INPUT_SCHEMA` (`items` + constraints 1..20) |
| `workers/lora_downloader/handler.py` | RunPod entry; flat return; `if __name__ == "__main__"` start |
| `workers/lora_downloader/tests/test_download.py` | Unit tests for download.py (no real network) |
| `workers/lora_downloader/tests/test_handler.py` | Handler tests (requires `runpod` installed) |
| `workers/lora_downloader/requirements.txt` | `runpod` — create **before** handler tests |
| `workers/lora_downloader/Dockerfile` | CPU slim image |
| `workers/lora_downloader/test_input.json` | Sample job input |
| `workers/lora_downloader/README.md` | Deploy invariants + API |
| `workers/lora_downloader/NOTICE` | RunPod pattern attribution |
| `workers/lora_downloader/LICENSES/RUNPOD-WORKER-SDXL-MIT.txt` | Copy from joycaption |
| `scripts/download_lora.py` | Operator CLI |
| `README.md` | Monorepo index rows |

### Constants (lock these names)

```python
VOLUME_ROOT = Path("/runpod-volume")
DEFAULT_LORA_DIR = VOLUME_ROOT / "krea2" / "loras"
MAX_ITEMS = 20
MAX_REDIRECTS = 5
HTTP_TIMEOUT_S = 600.0  # single socket timeout for urllib
USER_AGENT = "runpod-lora-downloader/1.0"
NOTE_RESTART = "Restart warm krea2 workers to pick up new LoRA files."
SAFETENSORS_SUFFIX = ".safetensors"
CHUNK_SIZE = 1024 * 1024
```

### Types

```python
@dataclass(frozen=True)
class NormalizedItem:
    model_version_id: str
    nsfw: bool
    filename: str | None  # None only if key absent; override present ⇒ str after normalize

@dataclass(frozen=True)
class ItemResult:
    model_version_id: str
    status: str  # "downloaded" | "skipped" | "failed"
    filename: str | None = None
    path: str | None = None
    bytes: int | None = None
    reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "model_version_id": self.model_version_id,
            "status": self.status,
        }
        if self.filename is not None:
            out["filename"] = self.filename
        if self.path is not None:
            out["path"] = self.path
        if self.bytes is not None:
            out["bytes"] = self.bytes
        if self.reason is not None:
            out["reason"] = self.reason
        if self.error is not None:
            out["error"] = self.error
        return out
```

### Interpreter convention

Host `python` may be 3.10; **target is 3.12**. Do **not** use global `pip install`.

| Phase | Interpreter |
|-------|-------------|
| Tasks 1–4 (stdlib only) | `python3.12` |
| Task 5+ (needs `runpod`) | `/tmp/runpod-lora-venv/bin/python` after venv create |

```bash
# Tasks 1–4:
PYTHONPATH=workers/lora_downloader python3.12 -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_download.py' -v

# Task 5 creates:
#   python3.12 -m venv /tmp/runpod-lora-venv
#   /tmp/runpod-lora-venv/bin/pip install -r workers/lora_downloader/requirements.txt

# Task 5+ full suite:
PYTHONPATH=workers/lora_downloader /tmp/runpod-lora-venv/bin/python -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_*.py' -v
```

---

### Task 1: Scaffold + `normalize_filename`

**Files:**
- Create: `workers/lora_downloader/download.py`
- Create: `workers/lora_downloader/tests/__init__.py`
- Create: `workers/lora_downloader/tests/test_download.py`

- [ ] **Step 1: Write failing tests**

```python
# workers/lora_downloader/tests/test_download.py
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
```

- [ ] **Step 2: Run — expect FAIL (import)**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.NormalizeFilenameTests -v
```

- [ ] **Step 3: Implement**

```python
# workers/lora_downloader/download.py
from __future__ import annotations

from pathlib import Path

SAFETENSORS_SUFFIX = ".safetensors"


class FilenameError(ValueError):
    """Invalid LoRA filename (safe client message)."""


def normalize_filename(name: object) -> str:
    if not isinstance(name, str):
        raise FilenameError("filename must be a string")
    name = name.strip()
    if not name:
        raise FilenameError("filename is empty")
    if "/" in name or "\\" in name:
        raise FilenameError("filename must not contain path separators")
    for ch in name:
        o = ord(ch)
        if o < 32 or o == 127:
            raise FilenameError("filename contains control characters")
    if not name.endswith(SAFETENSORS_SUFFIX):
        raise FilenameError("filename must end with .safetensors")
    stem = name[: -len(SAFETENSORS_SUFFIX)]
    if not stem or stem in (".", ".."):
        raise FilenameError("filename stem is invalid")
    return name
```

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.NormalizeFilenameTests -v
```

- [ ] **Step 5: Commit**

```bash
git add workers/lora_downloader/download.py workers/lora_downloader/tests/
git commit -m "feat(lora_downloader): add normalize_filename with safetensors rules"
```

---

### Task 2: `normalize_item` + `resolve_lora_dir`

**Files:**
- Modify: `workers/lora_downloader/download.py`
- Modify: `workers/lora_downloader/tests/test_download.py`

- [ ] **Step 1: Append failing tests**

```python
import os
import tempfile
from pathlib import Path
from unittest import mock

from download import (
    ItemNormalizeError,
    NormalizedItem,
    normalize_item,
    resolve_lora_dir,
)


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
        for raw in (0, -1, "", "12a", 1.5, None):
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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.NormalizeItemTests \
  workers.lora_downloader.tests.test_download.ResolveLoraDirTests -v
```

- [ ] **Step 3: Implement in `download.py` (append)**

```python
from dataclasses import dataclass
import os

VOLUME_ROOT = Path("/runpod-volume")
DEFAULT_LORA_DIR = VOLUME_ROOT / "krea2" / "loras"
MAX_ITEMS = 20


class ItemNormalizeError(ValueError):
    """Per-item validation failure."""


@dataclass(frozen=True)
class NormalizedItem:
    model_version_id: str
    nsfw: bool
    filename: str | None


def _normalize_model_version_id(value: object) -> str:
    if type(value) is bool:
        raise ItemNormalizeError("model_version_id must not be a boolean")
    if type(value) is int:
        if value <= 0:
            raise ItemNormalizeError("model_version_id must be a positive integer")
        return str(value)
    if isinstance(value, str):
        if not value.isdigit():
            raise ItemNormalizeError("model_version_id must be a positive digit string")
        as_int = int(value)
        if as_int <= 0:
            raise ItemNormalizeError("model_version_id must be a positive digit string")
        return str(as_int)
    raise ItemNormalizeError("model_version_id must be a positive int or digit string")


def normalize_item(raw: object) -> NormalizedItem:
    if not isinstance(raw, dict):
        raise ItemNormalizeError("item must be an object")
    if "model_version_id" not in raw:
        raise ItemNormalizeError("model_version_id is required")
    mid = _normalize_model_version_id(raw["model_version_id"])
    if "nsfw" not in raw:
        nsfw = True
    else:
        if type(raw["nsfw"]) is not bool:
            raise ItemNormalizeError("nsfw must be a boolean")
        nsfw = raw["nsfw"]
    filename: str | None = None
    if "filename" in raw:
        # Present key must be a str that passes normalize_filename (null → error).
        try:
            filename = normalize_filename(raw["filename"])
        except FilenameError as err:
            raise ItemNormalizeError(str(err)) from err
    return NormalizedItem(mid, nsfw, filename)


def _is_strict_child(path: Path, root: Path) -> bool:
    path_r = path.resolve()
    root_r = root.resolve()
    return root_r in path_r.parents


def resolve_lora_dir(env: dict[str, str] | None = None) -> Path:
    env_map = env if env is not None else os.environ
    raw = env_map.get("LORA_DIR", str(DEFAULT_LORA_DIR))
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_symlink():
        raise ValueError("LORA_DIR must not be a symlink")
    resolved = path.resolve()
    if not _is_strict_child(resolved, VOLUME_ROOT):
        raise ValueError("LORA_DIR must resolve under /runpod-volume")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("LORA_DIR must not be a symlink")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("LORA_DIR must be a directory")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
```

- [ ] **Step 4: Run — PASS**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.NormalizeItemTests \
  workers.lora_downloader.tests.test_download.ResolveLoraDirTests -v
```

- [ ] **Step 5: Commit**

```bash
git add workers/lora_downloader/download.py workers/lora_downloader/tests/test_download.py
git commit -m "feat(lora_downloader): normalize items and resolve LORA_DIR under volume"
```

---

### Task 3: HTTP download + atomic write (full tests, no `...`)

**Files:**
- Modify: `workers/lora_downloader/download.py`
- Modify: `workers/lora_downloader/tests/test_download.py`

Inject opener via `urlopen` callback:

```python
UrlOpen = Callable[..., Any]  # (Request, timeout=...) -> response-like
```

- [ ] **Step 1: Append full HTTP tests**

```python
import io
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from download import (
    FilenameError,
    ItemResult,
    NormalizedItem,
    download_to_lora_dir,
    extract_filename_from_response,
    resolve_api_url,
)


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
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.ResolveApiUrlTests \
  workers.lora_downloader.tests.test_download.ExtractFilenameTests \
  workers.lora_downloader.tests.test_download.DownloadToLoraDirTests -v
```

- [ ] **Step 3: Implement download helpers (full code)**

Append to `download.py` (**do not** re-add `from __future__ import annotations` —
it already exists at the top of the file from Task 1):

```python
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

MAX_REDIRECTS = 5
HTTP_TIMEOUT_S = 600.0
USER_AGENT = "runpod-lora-downloader/1.0"
NOTE_RESTART = "Restart warm krea2 workers to pick up new LoRA files."
CHUNK_SIZE = 1024 * 1024
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

UrlOpen = Callable[..., Any]


@dataclass(frozen=True)
class ItemResult:
    model_version_id: str
    status: str
    filename: str | None = None
    path: str | None = None
    bytes: int | None = None
    reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "model_version_id": self.model_version_id,
            "status": self.status,
        }
        if self.filename is not None:
            out["filename"] = self.filename
        if self.path is not None:
            out["path"] = self.path
        if self.bytes is not None:
            out["bytes"] = self.bytes
        if self.reason is not None:
            out["reason"] = self.reason
        if self.error is not None:
            out["error"] = self.error
        return out


def resolve_api_url(model_version_id: str, nsfw: bool) -> str:
    host = "civitai.red" if nsfw else "civitai.com"
    return f"https://{host}/api/download/models/{model_version_id}"


def _filename_from_content_disposition(value: str) -> str | None:
    # minimal parse: filename="..." or filename=...
    parts = value.split("filename=")
    if len(parts) < 2:
        return None
    rest = parts[1].strip()
    if rest.startswith('"'):
        end = rest.find('"', 1)
        if end == -1:
            return unquote(rest[1:])
        return unquote(rest[1:end])
    return unquote(rest.split(";")[0].strip())


def extract_filename_from_response(url: str, resp: Any) -> str:
    # Spec order: query disposition → Content-Disposition header → URL path
    query = parse_qs(urlparse(url).query)
    for key in ("response-content-disposition", "b2ContentDisposition"):
        values = query.get(key) or []
        if values:
            parsed = _filename_from_content_disposition(values[0])
            if parsed:
                return parsed
    cd = resp.getheader("Content-Disposition")
    if cd:
        parsed = _filename_from_content_disposition(cd)
        if parsed:
            return parsed
    path = urlparse(url).path
    base = path.rsplit("/", 1)[-1] if path else ""
    if base:
        return unquote(base)
    raise FilenameError("Unable to determine filename")


def _default_urlopen(req: urllib.request.Request, timeout: float = HTTP_TIMEOUT_S):
    class NoRedirect(urllib.request.HTTPErrorProcessor):
        def http_response(self, request, response):
            return response

        https_response = http_response

    opener = urllib.request.build_opener(NoRedirect)
    return opener.open(req, timeout=timeout)


def _failed(mid: str, error: str, filename: str | None = None) -> ItemResult:
    return ItemResult(
        model_version_id=mid, status="failed", filename=filename, error=error
    )


def _target_conflict(target: Path) -> str | None:
    """Return skip reason, fail message, or None if missing."""
    if not target.exists() and not target.is_symlink():
        return None
    # exists or dangling/broken symlink
    if target.is_symlink():
        return "fail:target is a symlink"
    if target.is_dir():
        return "fail:target is a directory"
    if target.is_file():
        return "skip"
    # FIFO, socket, device, etc.
    return "fail:target is not a regular file"


def download_to_lora_dir(
    item: NormalizedItem,
    lora_dir: Path,
    token: str,
    *,
    urlopen: UrlOpen | None = None,
) -> ItemResult:
    mid = item.model_version_id
    open_url = urlopen or _default_urlopen

    # Early skip/fail when filename override is known (no HTTP).
    if item.filename is not None:
        target = lora_dir / item.filename
        conflict = _target_conflict(target)
        if conflict == "skip":
            return ItemResult(
                model_version_id=mid,
                status="skipped",
                filename=item.filename,
                path=str(target),
                reason="already_exists",
            )
        if conflict and conflict.startswith("fail:"):
            return _failed(mid, conflict[5:], filename=item.filename)

    url = resolve_api_url(item.model_version_id, item.nsfw)
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    partial_path: str | None = None
    try:
        hops = 0
        while True:
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                # Always close the response (redirects + final + early returns).
                with open_url(req, timeout=HTTP_TIMEOUT_S) as resp:
                    status = getattr(resp, "status", None) or getattr(
                        resp, "code", None
                    )
                    if status in REDIRECT_STATUS:
                        if hops >= MAX_REDIRECTS:
                            return _failed(
                                mid, "too many redirects", filename=item.filename
                            )
                        loc = resp.getheader("Location")
                        if not loc:
                            return _failed(
                                mid,
                                "redirect missing Location",
                                filename=item.filename,
                            )
                        next_url = urljoin(url, loc)
                        if urlparse(next_url).scheme != "https":
                            return _failed(
                                mid,
                                "redirect must be HTTPS",
                                filename=item.filename,
                            )
                        url = next_url
                        headers = {"User-Agent": USER_AGENT}
                        hops += 1
                        continue

                    if status == 404:
                        return _failed(
                            mid, "File not found", filename=item.filename
                        )
                    if status != 200:
                        return _failed(
                            mid, f"HTTP {status}", filename=item.filename
                        )

                    try:
                        if item.filename is not None:
                            name = item.filename
                        else:
                            name = normalize_filename(
                                extract_filename_from_response(url, resp)
                            )
                    except FilenameError as err:
                        return _failed(mid, str(err), filename=item.filename)

                    target = lora_dir / name
                    conflict = _target_conflict(target)
                    if conflict == "skip":
                        return ItemResult(
                            model_version_id=mid,
                            status="skipped",
                            filename=name,
                            path=str(target),
                            reason="already_exists",
                        )
                    if conflict and conflict.startswith("fail:"):
                        return _failed(mid, conflict[5:], filename=name)

                    content_length = resp.getheader("Content-Length")
                    expected = (
                        int(content_length) if content_length is not None else None
                    )

                    fd, partial_path = tempfile.mkstemp(
                        prefix=".", suffix=".partial", dir=str(lora_dir)
                    )
                    downloaded = 0
                    try:
                        with os.fdopen(fd, "wb") as out:
                            while True:
                                chunk = resp.read(CHUNK_SIZE)
                                if not chunk:
                                    break
                                out.write(chunk)
                                downloaded += len(chunk)
                            out.flush()
                            os.fsync(out.fileno())
                        if expected is not None and downloaded != expected:
                            os.unlink(partial_path)
                            partial_path = None
                            return _failed(
                                mid,
                                "Content-Length mismatch",
                                filename=name,
                            )
                        conflict = _target_conflict(target)
                        if conflict == "skip":
                            os.unlink(partial_path)
                            partial_path = None
                            return ItemResult(
                                model_version_id=mid,
                                status="skipped",
                                filename=name,
                                path=str(target),
                                reason="already_exists",
                            )
                        if conflict and conflict.startswith("fail:"):
                            os.unlink(partial_path)
                            partial_path = None
                            return _failed(mid, conflict[5:], filename=name)
                        os.replace(partial_path, target)
                        partial_path = None
                        return ItemResult(
                            model_version_id=mid,
                            status="downloaded",
                            filename=name,
                            path=str(target),
                            bytes=downloaded,
                        )
                    finally:
                        if partial_path is not None and os.path.exists(
                            partial_path
                        ):
                            try:
                                os.unlink(partial_path)
                            except OSError:
                                pass
            except urllib.error.HTTPError as err:
                # HTTPError is a response; close it if open raised.
                try:
                    err.close()
                except Exception:
                    pass
                if err.code == 404:
                    return _failed(mid, "File not found", filename=item.filename)
                return _failed(mid, f"HTTP {err.code}", filename=item.filename)
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                reason = getattr(err, "reason", err)
                return _failed(mid, str(reason), filename=item.filename)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as err:
        if partial_path is not None and os.path.exists(partial_path):
            try:
                os.unlink(partial_path)
            except OSError:
                pass
        return _failed(mid, str(err), filename=item.filename)
```

**Notes for implementer (do not leave as TODOs in code):**

1. `Request.get_header("Authorization")` — urllib may capitalize; tests use `get_header("Authorization")` which works for headers set via dict on Request.
2. Counting redirects: after 5 successful redirect follows, the next 3xx fails (`hops >= MAX_REDIRECTS` with hops starting at 0 and incrementing each redirect).
3. Early override path must not call `urlopen`.

- [ ] **Step 4: Run all download tests — PASS**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_download.py' -v
```

- [ ] **Step 5: Commit**

```bash
git add workers/lora_downloader/download.py workers/lora_downloader/tests/test_download.py
git commit -m "feat(lora_downloader): CivitAI HTTPS download with redirect and atomic write"
```

---

### Task 4: `run_batch` with partial success (including exceptions)

**Files:**
- Modify: `workers/lora_downloader/download.py`
- Modify: `workers/lora_downloader/tests/test_download.py`

- [ ] **Step 1: Failing tests for batch**

```python
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
        from download import run_batch

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

    def test_network_error_then_success(self):
        from download import run_batch

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
        from download import ItemResult, NormalizedItem, run_batch

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
```

- [ ] **Step 2: Run — FAIL**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_download.RunBatchTests -v
```

- [ ] **Step 3: Implement `run_batch`**

```python
def _best_effort_id(raw: object) -> str:
    if isinstance(raw, dict) and "model_version_id" in raw:
        val = raw["model_version_id"]
        if type(val) is bool:
            return "?"
        if type(val) is int and val > 0:
            return str(val)
        if isinstance(val, str) and val.isdigit() and int(val) > 0:
            return str(int(val))
        return str(val)
    return "?"


def run_batch(
    items: list,
    *,
    token: str,
    lora_dir: Path,
    urlopen: UrlOpen | None = None,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for raw in items:
        try:
            item = normalize_item(raw)
        except ItemNormalizeError as err:
            results.append(
                ItemResult(
                    model_version_id=_best_effort_id(raw),
                    status="failed",
                    error=str(err),
                ).as_dict()
            )
            continue
        try:
            result = download_to_lora_dir(
                item, lora_dir, token, urlopen=urlopen
            )
        except Exception as err:  # noqa: BLE001 — item isolation
            result = ItemResult(
                model_version_id=item.model_version_id,
                status="failed",
                filename=item.filename,
                error=str(err),
            )
        results.append(result.as_dict())

    summary = {"downloaded": 0, "skipped": 0, "failed": 0}
    for row in results:
        status = row.get("status")
        if status in summary:
            summary[status] += 1
    return {
        "dest": str(lora_dir),
        "results": results,
        "summary": summary,
        "note": NOTE_RESTART,
    }


def get_civitai_token(env: dict[str, str] | None = None) -> str:
    env_map = env if env is not None else os.environ
    token = (env_map.get("CIVITAI_TOKEN") or "").strip()
    if not token:
        raise ValueError("missing CIVITAI_TOKEN")
    return token
```

- [ ] **Step 4: Run full download suite — PASS**

```bash
PYTHONPATH=workers/lora_downloader python3.12 -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_download.py' -v
```

- [ ] **Step 5: Commit**

```bash
git add workers/lora_downloader/download.py workers/lora_downloader/tests/test_download.py
git commit -m "feat(lora_downloader): run_batch partial success across items"
```

---

### Task 5: requirements + schemas + handler (runpod install first)

**Files:**
- Create: `workers/lora_downloader/requirements.txt`
- Create: `workers/lora_downloader/schemas.py`
- Create: `workers/lora_downloader/handler.py`
- Create: `workers/lora_downloader/tests/test_handler.py`

- [ ] **Step 1: Create requirements and task-local venv (Python 3.12)**

```text
# workers/lora_downloader/requirements.txt
runpod>=1.7.9
```

```bash
python3.12 -m venv /tmp/runpod-lora-venv
/tmp/runpod-lora-venv/bin/pip install -U pip
/tmp/runpod-lora-venv/bin/pip install -r workers/lora_downloader/requirements.txt
/tmp/runpod-lora-venv/bin/python -c "import runpod; print(runpod.__version__)"
```

Expected: version string prints (no ImportError).  
**Do not** `pip install` into the system/global interpreter.

- [ ] **Step 2: Write `schemas.py`**

```python
"""RunPod input validation schema for lora_downloader (job-level only)."""

from download import MAX_ITEMS

INPUT_SCHEMA = {
    "items": {
        "type": list,
        "required": True,
        "constraints": lambda items: (
            isinstance(items, list) and 1 <= len(items) <= MAX_ITEMS
        ),
    },
}
```

No separate `validate_items_list`. Nested item fields are **not** in schema.

- [ ] **Step 3: Write failing handler tests**

```python
# workers/lora_downloader/tests/test_handler.py
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from handler import handler


class HandlerTests(unittest.TestCase):
    def test_requires_input_key(self):
        out = handler({"items": [{"model_version_id": "1"}]})
        self.assertIn("error", out)

    def test_empty_items_error(self):
        out = handler({"input": {"items": []}})
        self.assertIn("error", out)

    def test_too_many_items_error(self):
        items = [{"model_version_id": str(i + 1)} for i in range(21)]
        out = handler({"input": {"items": items}})
        self.assertIn("error", out)

    def test_missing_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            # Ensure CIVITAI_TOKEN absent; LORA_DIR may still matter after token check
            out = handler({"input": {"items": [{"model_version_id": "1"}]}})
        self.assertIn("error", out)
        self.assertIn("CIVITAI_TOKEN", out["error"])

    def test_success_shape_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = Path(tmp) / "loras"
            lora.mkdir()
            env = {"CIVITAI_TOKEN": "tok", "LORA_DIR": str(lora)}
            fake_batch = {
                "dest": str(lora),
                "results": [
                    {
                        "model_version_id": "1",
                        "status": "downloaded",
                        "filename": "a.safetensors",
                    }
                ],
                "summary": {"downloaded": 1, "skipped": 0, "failed": 0},
                "note": "Restart warm krea2 workers to pick up new LoRA files.",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                with mock.patch("handler.resolve_lora_dir", return_value=lora):
                    with mock.patch(
                        "handler.run_batch", return_value=fake_batch
                    ) as rb:
                        out = handler(
                            {
                                "input": {
                                    "items": [
                                        {
                                            "model_version_id": "1",
                                            "filename": "a.safetensors",
                                        }
                                    ]
                                }
                            }
                        )
            self.assertEqual(out, fake_batch)
            self.assertNotIn("output", out)
            rb.assert_called_once()
            self.assertEqual(rb.call_args.kwargs["token"], "tok")
            self.assertEqual(rb.call_args.kwargs["lora_dir"], lora)
```

Handler imports: `from download import get_civitai_token, resolve_lora_dir, run_batch`.

- [ ] **Step 4: Implement `handler.py`**

```python
"""RunPod serverless handler — CivitAI LoRA downloader (CPU, volume write)."""

from __future__ import annotations

import logging
import os

import runpod
from runpod.serverless.utils.rp_validator import validate

from download import get_civitai_token, resolve_lora_dir, run_batch
from schemas import INPUT_SCHEMA

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("lora_downloader.handler")


def handler(job: dict) -> dict:
    if "input" not in job or not isinstance(job["input"], dict):
        return {"error": "job must contain an input object"}

    validated = validate(job["input"], INPUT_SCHEMA)
    if "errors" in validated:
        return {"error": str(validated["errors"])}

    payload = validated["validated_input"]
    items = payload["items"]

    try:
        token = get_civitai_token()
        lora_dir = resolve_lora_dir()
    except (ValueError, OSError) as err:
        return {"error": str(err)}

    # Flat payload — RunPod wraps as output.
    return run_batch(items, token=token, lora_dir=lora_dir)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
```

**Only** this start guard — not module-level unconditional start.

- [ ] **Step 5: Run handler + download tests — PASS**

```bash
PYTHONPATH=workers/lora_downloader /tmp/runpod-lora-venv/bin/python -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_*.py' -v
```

- [ ] **Step 6: Commit**

```bash
git add workers/lora_downloader/requirements.txt \
  workers/lora_downloader/schemas.py \
  workers/lora_downloader/handler.py \
  workers/lora_downloader/tests/test_handler.py
git commit -m "feat(lora_downloader): handler with schema bounds and flat return"
```

---

### Task 6: Dockerfile, NOTICE, test_input

**Files:**
- Create: `workers/lora_downloader/Dockerfile`
- Create: `workers/lora_downloader/test_input.json`
- Create: `workers/lora_downloader/NOTICE`
- Create: `workers/lora_downloader/LICENSES/RUNPOD-WORKER-SDXL-MIT.txt`

- [ ] **Step 1: Dockerfile**

```dockerfile
# CivitAI LoRA downloader — thin RunPod serverless CPU worker
# Dockerfile path: workers/lora_downloader/Dockerfile
# Build context: repository root

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY workers/lora_downloader/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY workers/lora_downloader/schemas.py \
     workers/lora_downloader/handler.py \
     workers/lora_downloader/download.py \
     workers/lora_downloader/test_input.json \
     workers/lora_downloader/NOTICE \
     /app/
COPY workers/lora_downloader/LICENSES /app/LICENSES

ENV LORA_DIR=/runpod-volume/krea2/loras \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

CMD ["python", "-u", "/app/handler.py"]
```

- [ ] **Step 2: test_input.json**

```json
{
  "input": {
    "items": [
      {
        "model_version_id": "46846",
        "filename": "example.safetensors",
        "nsfw": true
      }
    ]
  }
}
```

- [ ] **Step 3: NOTICE**

```text
Third-party software notices for workers/lora_downloader

RunPod worker-sdxl
------------------

Portions of handler.py (serverless validation / entrypoint pattern) are adapted
from:

  runpod-workers/worker-sdxl
  https://github.com/runpod-workers/worker-sdxl

Copyright (c) 2025 RunPod

The adapted portions are licensed under the MIT License. A copy of that
license is provided in LICENSES/RUNPOD-WORKER-SDXL-MIT.txt.

CivitAI
-------

This worker downloads files using CivitAI's public download API
(https://civitai.com / https://civitai.red) with an operator-provided API token.
It does not vendor or copy third-party download utilities (including any
GPL-licensed clients). HTTP client code in download.py is original to this
repository.

LoRA weights are NOT included in this repository.
```

- [ ] **Step 4: Copy license file**

```bash
mkdir -p workers/lora_downloader/LICENSES
cp workers/joycaption/LICENSES/RUNPOD-WORKER-SDXL-MIT.txt \
  workers/lora_downloader/LICENSES/RUNPOD-WORKER-SDXL-MIT.txt
```

- [ ] **Step 5: Docker build check (must succeed)**

From monorepo root (requires Docker daemon):

```bash
docker build -f workers/lora_downloader/Dockerfile -t runpod-lora-downloader:test .
```

Expected: build completes with exit code 0 (image tagged `runpod-lora-downloader:test`).

Optional smoke that CMD is importable:

```bash
docker run --rm runpod-lora-downloader:test \
  python -c "import download, handler, schemas; print('ok')"
```

Expected: prints `ok` (does not start serverless because `__name__ != "__main__"`).

- [ ] **Step 6: Commit**

```bash
git add workers/lora_downloader/Dockerfile \
  workers/lora_downloader/test_input.json \
  workers/lora_downloader/NOTICE \
  workers/lora_downloader/LICENSES/
git commit -m "chore(lora_downloader): add CPU Dockerfile, NOTICE, test_input"
```

---

### Task 7: CLI `scripts/download_lora.py` (complete)

**Files:**
- Create: `scripts/download_lora.py`
- Create: `workers/lora_downloader/tests/test_cli_exit.py` (tests pure helpers via import)

CLI helpers live in the script. Tests load it via `importlib.util.spec_from_file_location`
(see `test_cli_exit.py` below).

```python
# scripts/download_lora.py  — full file
#!/usr/bin/env python3
"""Download CivitAI LoRAs via the lora_downloader RunPod endpoint.

Env:
  RUNPOD_API_KEY   required
  ENDPOINT_ID      required unless --endpoint-id is passed

Examples:
  export RUNPOD_API_KEY=rp_xxx
  export ENDPOINT_ID=abc123
  python scripts/download_lora.py 46846
  python scripts/download_lora.py 46846 999 --filename 46846=my.safetensors
  python scripts/download_lora.py 46846 --sfw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 2.0


def exit_code_for_job_result(result: dict[str, Any]) -> int:
    if result.get("status") != "COMPLETED":
        return 1
    output = result.get("output")
    if not isinstance(output, dict):
        return 1
    if output.get("error"):
        return 1
    summary = output.get("summary") or {}
    try:
        failed = int(summary.get("failed") or 0)
    except (TypeError, ValueError):
        return 1
    if failed > 0:
        return 1
    return 0


def build_job_input(
    version_ids: list[str],
    *,
    filenames: dict[str, str],
    sfw: bool,
) -> dict[str, Any]:
    items = []
    for vid in version_ids:
        item: dict[str, Any] = {
            "model_version_id": vid,
            "nsfw": not sfw,
        }
        if vid in filenames:
            item["filename"] = filenames[vid]
        items.append(item)
    return {"items": items}


def parse_filename_args(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"error: --filename must be ID=name.safetensors, got {raw!r}")
        vid, name = raw.split("=", 1)
        vid = vid.strip()
        name = name.strip()
        if not vid or not name:
            raise SystemExit(f"error: bad --filename {raw!r}")
        out[vid] = name
    return out


def require_api_key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        print("error: set RUNPOD_API_KEY", file=sys.stderr)
        raise SystemExit(1)
    return value


def require_endpoint_id(cli_value: str | None) -> str:
    value = (cli_value or os.environ.get("ENDPOINT_ID") or "").strip()
    if not value:
        print(
            "error: set ENDPOINT_ID env or pass --endpoint-id",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def api_request(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        print(f"error: HTTP {err.code}: {detail[:500]}", file=sys.stderr)
        raise SystemExit(1) from err
    except urllib.error.URLError as err:
        print(f"error: request failed: {err.reason}", file=sys.stderr)
        raise SystemExit(1) from err
    if not raw:
        return {}
    return json.loads(raw)


def submit_job(endpoint_id: str, api_key: str, job_input: dict[str, Any]) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    resp = api_request("POST", url, api_key, {"input": job_input}, timeout=120.0)
    job_id = resp.get("id")
    if not job_id:
        print(f"error: no job id: {resp}", file=sys.stderr)
        raise SystemExit(1)
    print(f"job id: {job_id}", file=sys.stderr)
    return str(job_id)


def poll_job(
    endpoint_id: str, api_key: str, job_id: str, *, interval: float
) -> dict[str, Any]:
    url = f"{API_BASE}/{endpoint_id}/status/{job_id}"
    while True:
        resp = api_request("GET", url, api_key, timeout=30.0)
        status = resp.get("status") or "?"
        print(f"status={status}", file=sys.stderr, flush=True)
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return resp
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version_ids",
        nargs="+",
        help="CivitAI model version id(s)",
    )
    parser.add_argument(
        "--filename",
        action="append",
        default=[],
        help="Map version id to filename: ID=name.safetensors (repeatable)",
    )
    parser.add_argument(
        "--sfw",
        action="store_true",
        help="Use civitai.com (nsfw=false) for all items",
    )
    parser.add_argument(
        "--endpoint-id",
        default=None,
        help="RunPod endpoint id (else ENDPOINT_ID env)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
    )
    args = parser.parse_args(argv)

    api_key = require_api_key()
    endpoint_id = require_endpoint_id(args.endpoint_id)
    filenames = parse_filename_args(args.filename)
    job_input = build_job_input(
        args.version_ids, filenames=filenames, sfw=args.sfw
    )
    job_id = submit_job(endpoint_id, api_key, job_input)
    result = poll_job(
        endpoint_id, api_key, job_id, interval=args.poll_interval
    )
    code = exit_code_for_job_result(result)
    output = result.get("output")
    if isinstance(output, dict):
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if code == 0:
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: CLI unit tests**

```python
# workers/lora_downloader/tests/test_cli_exit.py
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "download_lora.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("download_lora_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class CliHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = _load_cli()

    def test_exit_codes(self):
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"summary": {"failed": 0}}}
            ),
            0,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"summary": {"failed": 1}}}
            ),
            1,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result({"status": "FAILED"}),
            1,
        )
        self.assertEqual(
            self.cli.exit_code_for_job_result(
                {"status": "COMPLETED", "output": {"error": "missing token"}}
            ),
            1,
        )

    def test_build_job_input_sfw(self):
        inp = self.cli.build_job_input(
            ["1", "2"], filenames={"1": "a.safetensors"}, sfw=True
        )
        self.assertEqual(
            inp,
            {
                "items": [
                    {
                        "model_version_id": "1",
                        "nsfw": False,
                        "filename": "a.safetensors",
                    },
                    {"model_version_id": "2", "nsfw": False},
                ]
            },
        )

    def test_build_job_input_nsfw_default(self):
        inp = self.cli.build_job_input(["9"], filenames={}, sfw=False)
        self.assertTrue(inp["items"][0]["nsfw"])
```

- [ ] **Step 3: Run tests — PASS**

```bash
# CLI helper tests need no runpod; python3.12 is enough
PYTHONPATH=workers/lora_downloader python3.12 -m unittest \
  workers.lora_downloader.tests.test_cli_exit -v
```

- [ ] **Step 4: Commit**

```bash
git add scripts/download_lora.py workers/lora_downloader/tests/test_cli_exit.py
git commit -m "feat(lora_downloader): add download_lora CLI client"
```

---

### Task 8: README + monorepo index

**Files:**
- Create: `workers/lora_downloader/README.md`
- Modify: `README.md`

- [ ] **Step 1: Create `workers/lora_downloader/README.md` with this full text**

~~~~markdown
# RunPod worker — lora_downloader

Thin **CPU** serverless worker: batch-download CivitAI LoRA files onto the
shared Network Volume for [krea2](../krea2/) runtime adapters.

| | |
|--|--|
| **Compute** | Serverless **CPU** (no GPU, no torch) |
| **Write path** | `$LORA_DIR` (default `/runpod-volume/krea2/loras`) |
| **Auth** | `CIVITAI_TOKEN` endpoint secret |
| **Design** | [lora_downloader design](../../docs/superpowers/specs/2026-08-03-lora-downloader-design.md) |

HTTP client code is **original** (stdlib `urllib`). This repo does **not**
vendor [civitai-downloader](https://github.com/ashleykleynhans/civitai-downloader)
(GPL-3.0).

## Layout

```
workers/lora_downloader/
├── Dockerfile
├── handler.py
├── download.py
├── schemas.py
├── requirements.txt
├── test_input.json
├── NOTICE
├── LICENSES/
└── tests/
```

## Deploy invariants (required)

1. **Serverless CPU** endpoint (not GPU).
2. **Same network volume + datacenter** as the krea2 endpoint.
3. Volume mounts at `/runpod-volume`.
4. **`workersMax = 1`** — this endpoint is the **only writer** of `$LORA_DIR`.
   Do not raise max workers (RunPod warns about concurrent volume writes).
5. Secret: `CIVITAI_TOKEN=<civitai api key>`.
6. Optional env: `LORA_DIR=/runpod-volume/krea2/loras` (must resolve **under**
   hard-coded `/runpod-volume`; must not be a symlink directory).
7. Idle timeout can be low; execution timeout must allow large multi-file batches.

### GitHub / image

- Dockerfile path: `workers/lora_downloader/Dockerfile`
- Build context: repository root

```bash
docker build -f workers/lora_downloader/Dockerfile -t runpod-lora-downloader .
```

## Network volume

```text
/runpod-volume/krea2/loras/
  my_style.safetensors   # krea2 LoRA id = my_style
```

After new files appear, **restart warm krea2 workers** so they re-scan the
allowlist (krea2 catalogs stems only at process start).

## Env

| Variable | Default | Meaning |
|----------|---------|---------|
| `CIVITAI_TOKEN` | (required) | CivitAI API key |
| `LORA_DIR` | `/runpod-volume/krea2/loras` | Sole write directory |
| `LOG_LEVEL` | `INFO` | Logging |

## API

### Input

```json
{
  "input": {
    "items": [
      {
        "model_version_id": "46846",
        "filename": "my_style.safetensors",
        "nsfw": true
      },
      {
        "model_version_id": "99999"
      }
    ]
  }
}
```

| Field | Level | Required | Default | Notes |
|-------|-------|:--------:|---------|-------|
| `items` | job | yes | — | list, length 1…20 |
| `items[].model_version_id` | item | yes | — | positive int or digit string; **not** bool |
| `items[].filename` | item | no | from CivitAI response | if key present, must be a string |
| `items[].nsfw` | item | no | `true` | **strict bool**; `true` → civitai.red |

No job-level `dest` field.

### Filename rules

Final name (override or extracted) must:

- end with `.safetensors` (case-sensitive)
- have a non-empty stem that is not `.` or `..`
- contain no `/`, `\`, or control characters
- allow Unicode and internal spaces

### Conflict policy

| Target path state | Result |
|-------------------|--------|
| missing | download |
| regular file | `skipped` / `already_exists` |
| symlink, directory, FIFO, device, etc. | `failed` |

If `filename` override is set and the target already exists as a regular file,
the worker **skips without calling CivitAI**.

### Success output (flat handler return)

RunPod wraps the handler return value as `output`. The handler returns:

```json
{
  "dest": "/runpod-volume/krea2/loras",
  "results": [
    {
      "model_version_id": "46846",
      "filename": "my_style.safetensors",
      "status": "downloaded",
      "bytes": 123,
      "path": "/runpod-volume/krea2/loras/my_style.safetensors"
    }
  ],
  "summary": { "downloaded": 1, "skipped": 0, "failed": 0 },
  "note": "Restart warm krea2 workers to pick up new LoRA files."
}
```

Per-item failures do not fail the whole job; check `summary.failed`.

### Job-level errors

Missing/invalid `input.items`, missing `CIVITAI_TOKEN`, or invalid `LORA_DIR`
return `{"error": "..."}`.

## CLI

```bash
export RUNPOD_API_KEY=...
export ENDPOINT_ID=...   # required (no hardcoded default)

python scripts/download_lora.py 46846
python scripts/download_lora.py 46846 --filename 46846=my_style.safetensors
python scripts/download_lora.py 46846 --sfw
```

- `--sfw` sets `nsfw: false` for all items
- Prints unwrapped worker `output` JSON to stdout
- Exit code **non-zero** if job error or `summary.failed > 0`

## License

See [NOTICE](NOTICE) and [LICENSES/](LICENSES/). RunPod handler patterns are
adapted under MIT. LoRA weights and CivitAI ToS are the operator's responsibility;
weights are not stored in this repository.
~~~~

- [ ] **Step 2: Update root `README.md`**

In the Workers table add a row:

```markdown
| [`workers/lora_downloader`](workers/lora_downloader/) | CivitAI LoRA download to network volume (CPU) | MVP |
```

In the Remote test clients / scripts table add:

```markdown
| [`scripts/download_lora.py`](scripts/download_lora.py) | set `ENDPOINT_ID` | version ids → volume |
```

- [ ] **Step 3: Full suite (venv interpreter)**

```bash
PYTHONPATH=workers/lora_downloader /tmp/runpod-lora-venv/bin/python -m unittest discover \
  -s workers/lora_downloader/tests -p 'test_*.py' -v
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add workers/lora_downloader/README.md README.md
git commit -m "docs(lora_downloader): README and monorepo index"
```

---

## Spec coverage checklist

| Spec / review requirement | Task |
|---------------------------|------|
| Original stdlib HTTP, no GPL | 3 |
| Flat handler return | 5 |
| Job `items` 1..20 via schema constraints | 5 |
| Item loop normalize → failed | 2, 4 |
| Exception isolation (URLError etc.) + batch continues | 3, 4 |
| Override exists → skip **before** HTTP | 3 |
| Non-regular target (symlink/dir/fifo/other) → failed | 3 |
| `filename: null` → item error | 2 |
| Strict `job["input"]` | 5 |
| `ValueError` + `OSError` on resolve | 5 |
| Single `HTTP_TIMEOUT_S` | 3 |
| Bearer only first hop; max 5 HTTPS redirects | 3 |
| Response closed via `with open_url(...)` | 3 |
| Filename precedence query → header → path | 3 |
| Content-Length verify | 3 |
| tempfile atomic write | 3 |
| `run_batch` catches unexpected `Exception` (RuntimeError test) | 4 |
| python3.12 venv + requirements before handler tests; `if __name__ == "__main__"` | 5 |
| CPU Dockerfile + `docker build` check | 6 |
| CLI full: `--sfw`, unwrap, exit codes, required ENDPOINT_ID | 7 |
| Full README text + `workersMax=1` | 8 |

## Out of scope (do not implement)

- job `dest`, overwrite, list/delete, S3 path, safetensors magic, GPU image, GPL copy, `validate_items_list`, dual connect/read timeouts

---

## Plan revision notes

- Removed all `...` stubs in HTTP/batch/CLI; full test classes and implementation code.
- Partial success: network errors inside downloader **and** unexpected
  `RuntimeError` from `download_to_lora_dir` via `run_batch` isolation test.
- Filename override skip/fail before any `urlopen`.
- No second `from __future__` in Task 3 append block.
- Every HTTP response uses `with open_url(...) as resp:`.
- Task-local `python3.12` venv at `/tmp/runpod-lora-venv` (no global pip).
- `docker build` verification after Dockerfile task.
- Full README body in Task 8; filename extract order query → header → path.
- Spec alignment: null filename error; strict input key; catch `OSError`;
  non-regular targets failed; single `HTTP_TIMEOUT_S`; schema constraints only.
