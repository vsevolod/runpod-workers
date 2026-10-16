"""Start downloads before importing the worker's ML runtime."""

from __future__ import annotations

import argparse
import os
import runpy
import signal
import sys
from pathlib import Path

# Support execution as a script in both the repo and the standalone image.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.prefetch import PrefetchSession


def run_worker(app_dir: Path, download_args: list[str]) -> None:
    app_dir = app_dir.resolve()
    command = [sys.executable, "-u", str(app_dir / "download_weights.py"), *download_args]
    with PrefetchSession(command) as downloads:
        os.environ["LOCAL_FILES_ONLY"] = "1"
        sys.path.insert(0, str(app_dir))
        sys.argv = [str(app_dir / "handler.py")]
        runpy.run_path(sys.argv[0], run_name="__main__")
        downloads.wait("all")


def _terminate(signum, frame):
    # Let the session's finally block reap the downloader on container shutdown.
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--download-arg", action="append", default=[])
    args = parser.parse_args(argv)
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        run_worker(args.app_dir, args.download_arg)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
