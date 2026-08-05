#!/usr/bin/env python3
"""Phase 1 headless vertical slice:

  API JSON → POST /prompt → /history/{id} → SaveVideo metadata → MP4 → audio check

Does NOT pick largest .mp4. Requires running ComfyUI (≥ v0.30.0 pin) + four weights.

    python tools/headless_t2v_smoke.py \\
        --workflow workflows/t2va_api.json \\
        --comfy-url http://127.0.0.1:8188 \\
        --output-dir /path/to/ComfyUI/output
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

# Allow import of sibling package when run from worker root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow import (  # noqa: E402
    SAVE_VIDEO_NODE,
    inject_product,
    load_workflow,
)


def resolve_savevideo_path(history_entry: dict, output_dir: Path) -> Path:
    outputs = history_entry.get("outputs") or {}
    node = outputs.get(SAVE_VIDEO_NODE)
    if not isinstance(node, dict):
        raise RuntimeError(
            f"SaveVideo node {SAVE_VIDEO_NODE!r} missing; outputs keys={list(outputs)}"
        )
    images = node.get("images")
    if not images:
        raise RuntimeError(
            f"SaveVideo {SAVE_VIDEO_NODE}: no images[] (keys={list(node.keys())})"
        )
    meta = images[0]
    filename = meta["filename"]
    subfolder = meta.get("subfolder") or ""
    path = output_dir / subfolder / filename if subfolder else output_dir / filename
    return path


def has_audio_stream(mp4: Path) -> bool:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return "audio" in (proc.stdout or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workflow",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workflows" / "t2va_api.json",
    )
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os_env_output()),
    )
    ap.add_argument("--prompt", default="A quiet desk lamp, soft room tone, 2s still life.")
    ap.add_argument("--width", type=int, default=864)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    base = args.comfy_url.rstrip("/")
    wf = inject_product(
        load_workflow(args.workflow),
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        duration=args.duration,
        seed=args.seed,
    )

    client_id = str(uuid.uuid4())
    t0 = time.time()
    r = requests.post(
        f"{base}/prompt",
        json={"prompt": wf, "client_id": client_id},
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("error") or body.get("node_errors"):
        print(json.dumps(body, indent=2), file=sys.stderr)
        return 1
    prompt_id = body["prompt_id"]
    print(f"prompt_id={prompt_id}")

    history = None
    while time.time() - t0 < args.timeout:
        h = requests.get(f"{base}/history/{prompt_id}", timeout=60).json()
        if prompt_id in h and h[prompt_id].get("outputs"):
            history = h[prompt_id]
            break
        time.sleep(1.0)
    if history is None:
        print("timeout waiting for history", file=sys.stderr)
        return 1

    # Document exact path for PINS
    print("SaveVideo raw outputs:")
    print(json.dumps(history.get("outputs", {}).get(SAVE_VIDEO_NODE), indent=2))

    mp4 = resolve_savevideo_path(history, args.output_dir)
    if not mp4.is_file() or mp4.stat().st_size <= 0:
        print(f"missing/empty mp4: {mp4}", file=sys.stderr)
        return 1
    print(f"mp4={mp4} size={mp4.stat().st_size}")
    if not has_audio_stream(mp4):
        print("ffprobe: no audio stream", file=sys.stderr)
        return 1
    print(f"ok: MP4 with audio in {time.time() - t0:.1f}s")
    print(f"canvas={args.width}x{args.height} duration={args.duration}")
    return 0


def os_env_output() -> str:
    import os

    return os.environ.get("COMFY_OUTPUT_DIR", "output")


if __name__ == "__main__":
    raise SystemExit(main())
