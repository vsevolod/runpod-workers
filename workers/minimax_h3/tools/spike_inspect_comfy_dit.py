#!/usr/bin/env python3
"""G0 inspect. Exit 0 only if compatible_with_stock_diffusers."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from h3_infer.comfy_dit_g0 import classify_dit_file, read_safetensors_header


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    args = p.parse_args(argv)
    if not args.path.is_file():
        print(f"missing: {args.path}", file=sys.stderr)
        return 2
    r = classify_dit_file(args.path)
    print(
        json.dumps(
            {
                "path": str(args.path),
                "verdict": r.verdict,
                "compatible_with_stock_diffusers": r.compatible_with_stock_diffusers,
                "reasons": list(r.reasons),
                "notes": list(r.notes),
                "num_keys": len(read_safetensors_header(args.path)),
            },
            indent=2,
        )
    )
    return 0 if r.compatible_with_stock_diffusers else 1


if __name__ == "__main__":
    raise SystemExit(main())
