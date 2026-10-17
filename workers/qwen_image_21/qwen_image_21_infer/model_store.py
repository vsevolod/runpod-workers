"""Complete, request-selected snapshots on container disk."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

workers_dir = Path(__file__).resolve().parents[2]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.prefetch import PrefetchSession

from .runtime_config import (
    TEXTURE_FIX_VAE_ID, resolve_model_directory, resolve_model_id,
    resolve_vae_directory, resolve_vae_variant,
)

MANIFEST = ".qwen-complete.json"


def _read_manifest(root: Path):
    try:
        data = json.loads((root / MANIFEST).read_text())
        if data.get("format") != 1 or not isinstance(data.get("files"), list) or not data["files"]:
            return None
        for item in data["files"]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts or type(item["size"]) is not int or item["size"] <= 0:
                return None
        return data
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def snapshot_ready(directory, *, model_id, model_variant, profile=None):
    root = Path(directory)
    data = _read_manifest(root)
    if data is None or any(data.get(key) != value for key, value in (
        ("model_id", model_id), ("model_variant", model_variant), ("profile", profile),
    )):
        return False
    try:
        return all(not (root / item["path"]).is_symlink()
                   and (root / item["path"]).is_file()
                   and (root / item["path"]).stat().st_size == item["size"]
                   for item in data["files"])
    except OSError:
        return False


def remove_incomplete_files(directory):
    root = Path(directory)
    data = _read_manifest(root)
    if data is None:
        return
    for item in data["files"]:
        path = root / item["path"]
        try:
            if path.is_file() and path.stat().st_size != item["size"]:
                path.unlink()
        except FileNotFoundError:
            pass


def write_manifest(directory, *, model_id, model_variant, profile=None, revision=None):
    root = Path(directory)
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts) or relative.parts[0] in {"loras", "lora_cache", "texture_fix_vae"}:
            continue
        if path.is_file() and not path.is_symlink():
            files.append({"path": relative.as_posix(), "size": path.stat().st_size})
    if not files or any(item["size"] <= 0 for item in files):
        raise ValueError(f"Cannot publish an empty/incomplete model snapshot: {root}")
    payload = dict(format=1, model_id=model_id, model_variant=model_variant,
                   profile=profile, revision=revision, files=files)
    pending = root / (MANIFEST + ".tmp")
    pending.write_text(json.dumps(payload, indent=2))
    pending.replace(root / MANIFEST)


@contextmanager
def prefetch_model(model_variant):
    root = resolve_model_directory(model_variant)
    model_id = resolve_model_id(model_variant)
    vae = resolve_vae_directory()
    profile = resolve_vae_variant()
    complete = snapshot_ready(root, model_id=model_id, model_variant=model_variant, profile=profile)
    vae_complete = True
    if profile == "texture-fix":
        vae_complete = snapshot_ready(vae, model_id=TEXTURE_FIX_VAE_ID, model_variant="vae")
        complete = complete and vae_complete
    if complete:
        yield None
        return
    command = [sys.executable, "-u", str(Path(__file__).parents[1] / "download_weights.py"),
               "--model-variant", model_variant, "--output", str(root), "--vae-dir", str(vae)]
    with PrefetchSession(command) as pending:
        pending.reload_vae = not vae_complete
        yield pending
