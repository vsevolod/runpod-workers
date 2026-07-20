#!/usr/bin/env python3
"""Extract first PNG from a RunPod runsync/status JSON into the current directory."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_path",
        nargs="?",
        default="/tmp/rp-lora.json",
        help="Path to RunPod response JSON (default: /tmp/rp-lora.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.png",
        help="Output filename in the current working directory (default: output.png)",
    )
    args = parser.parse_args()

    src = Path(args.json_path)
    if not src.is_file():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 1

    data = json.loads(src.read_text())
    out = data.get("output") or data
    if not isinstance(out, dict):
        print(f"error: unexpected JSON shape: {type(out).__name__}", file=sys.stderr)
        return 1

    url = None
    images = out.get("images")
    if isinstance(images, list) and images:
        url = images[0]
    if not url:
        url = out.get("image_url")
    if not isinstance(url, str) or not url:
        print("error: no images/image_url in response", file=sys.stderr)
        return 1

    match = re.match(r"data:image/[^;]+;base64,(.+)$", url, re.DOTALL)
    b64 = match.group(1) if match else url
    try:
        raw = base64.b64decode(b64, validate=False)
    except Exception as err:
        print(f"error: base64 decode failed: {err}", file=sys.stderr)
        return 1

    dest = Path.cwd() / args.output
    dest.write_bytes(raw)
    print(f"wrote {dest} ({len(raw)} bytes)")
    if "loras" in out:
        print(f"loras: {out['loras']}")
    if "seed" in out:
        print(f"seed: {out['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
