"""Hub transfers, local cache setup and timing shared by model bootstrappers.

Import Hugging Face lazily, after the caller has selected its cache directory.
Model selection and validation belong to each worker's download_weights.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path


def add_download_arguments(parser: argparse.ArgumentParser, *, default_output: str) -> None:
    parser.add_argument(
        "--output", default=os.environ.get("MODEL_DIR", default_output),
        help="Model directory (default: MODEL_DIR or %(default)s)",
    )
    parser.add_argument(
        "--hf-home", default=os.environ.get("HF_HOME", ""),
        help="Optional local HF cache root; otherwise use the default HF cache.",
    )
    parser.add_argument(
        "--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="Hugging Face token (or set HF_TOKEN)",
    )


class ModelDownloader:
    def __init__(self, output: str | Path, *, hf_home: str | Path | None = None, token: str | None = None):
        self.output = Path(output).expanduser().resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.cache_dir = None
        if hf_home:
            home = Path(hf_home).expanduser().resolve()
            self.cache_dir = home / "hub"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(home)
            os.environ["HF_HUB_CACHE"] = str(self.cache_dir)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir)
            print(f"HF_HOME={home}")

    def __enter__(self):
        self.started = time.monotonic()
        print(f"stage=model_download begin path={self.output}", flush=True)
        return self

    def __exit__(self, exc_type, exc, traceback):
        status = "failed" if exc_type is not None else "done"
        print(f"stage=model_download {status} elapsed_s={time.monotonic() - self.started:.2f}", flush=True)

    def _hub_options(self) -> dict:
        options = {"token": self.token}
        if self.cache_dir is not None:
            # Explicit cache_dir also works if another caller imported the Hub
            # before this session selected HF_HOME.
            options["cache_dir"] = str(self.cache_dir)
        return options

    def snapshot(self, repo_id: str, *, local_dir: str | Path | None = None,
                 allow_patterns: list[str] | None = None) -> Path:
        from huggingface_hub import snapshot_download

        options = self._hub_options()
        if local_dir is not None:
            options["local_dir"] = str(local_dir)
        if allow_patterns is not None:
            options["allow_patterns"] = allow_patterns
        print(f"Snapshot {repo_id} …", flush=True)
        path = Path(snapshot_download(repo_id=repo_id, **options))
        print(f"  -> {path}", flush=True)
        return path

    def file(self, repo_id: str, filename: str) -> Path:
        from huggingface_hub import hf_hub_download

        print(f"Downloading {repo_id}/{filename} …", flush=True)
        path = Path(hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(self.output), **self._hub_options(),
        ))
        print(f"  -> {path}", flush=True)
        return path


def link_or_copy(source: Path, destination: Path) -> None:
    """Expose a downloaded nested file under a flat name without doubling disk use."""
    if destination.is_file():
        return
    try:
        destination.hardlink_to(source)
        print(f"  hardlink {destination} -> {source}")
    except OSError:
        shutil.copy2(source, destination)
        print(f"  copied {source} -> {destination}")
