from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urljoin, urlparse

SAFETENSORS_SUFFIX = ".safetensors"
VOLUME_ROOT = Path("/runpod-volume")
DEFAULT_LORA_DIR = VOLUME_ROOT / "krea2" / "loras"
MAX_ITEMS = 20
MAX_REDIRECTS = 5
HTTP_TIMEOUT_S = 600.0
USER_AGENT = "runpod-lora-downloader/1.0"
NOTE_RESTART = "Restart warm krea2 workers to pick up new LoRA files."
CHUNK_SIZE = 1024 * 1024
REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

UrlOpen = Callable[..., Any]


class FilenameError(ValueError):
    """Invalid LoRA filename (safe client message)."""


class ItemNormalizeError(ValueError):
    """Per-item validation failure."""


@dataclass(frozen=True)
class NormalizedItem:
    model_version_id: str
    nsfw: bool
    filename: str | None


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
