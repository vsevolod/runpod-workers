#!/usr/bin/env python3
"""G4: meta-init + load Comfy int8 DiT + real forward (fail-closed).

Exit codes:
  0 — forward ran and produced finite tensor(s)
  1 — G0/G1/G2/G3/G4 failure, missing files, or forward not wired
  2 — usage / missing path
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch


def _collect_tensors(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        found: list[torch.Tensor] = []
        for item in out:
            found.extend(_collect_tensors(item))
        return found
    if isinstance(out, dict):
        found = []
        for item in out.values():
            found.extend(_collect_tensors(item))
        return found
    # diffusers ModelOutput-like
    if hasattr(out, "sample"):
        return _collect_tensors(out.sample)
    if hasattr(out, "to_tuple"):
        return _collect_tensors(out.to_tuple())
    return []


def _build_dummy_kwargs(model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    """Construct legal dummy inputs for MiniMaxH3Transformer3DModel.forward.

    Inspects the installed pin signature. If required args cannot be satisfied,
    raises RuntimeError (caller exits 1 — never claim G4 PASS).
    """
    import inspect

    sig = inspect.signature(model.forward)
    params = sig.parameters
    cfg = getattr(model, "config", None)
    hidden = int(getattr(cfg, "hidden_size", 5376) if cfg is not None else 5376)
    in_ch = int(getattr(cfg, "in_channels", 24) if cfg is not None else 24)
    audio_ch = int(getattr(cfg, "audio_in_channels", 32) if cfg is not None else 32)

    # Minimal latent geometry (tiny spatial/temporal for smoke).
    b, t, h, w = 1, 5, 4, 4
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    kwargs: dict[str, Any] = {}
    # Common names in video DiT ports — probe and fill what the signature accepts.
    candidates: dict[str, Any] = {
        "hidden_states": torch.randn(b, in_ch, t, h, w, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(b, 8, getattr(cfg, "text_dim", 5120) if cfg else 5120, device=device, dtype=dtype),
        "timestep": torch.tensor([500], device=device, dtype=torch.long),
        "timestep_audio": torch.tensor([500], device=device, dtype=torch.long),
        "audio_hidden_states": torch.randn(b, audio_ch, t * 4, device=device, dtype=dtype),
        "return_dict": False,
    }

    for name, default in candidates.items():
        if name in params:
            kwargs[name] = default

    # Require at least hidden_states + timestep style args if present in signature.
    required = [
        n
        for n, p in params.items()
        if p.default is inspect.Parameter.empty
        and n not in ("self", "kwargs")
        and p.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]
    missing = [n for n in required if n not in kwargs]
    if missing:
        raise RuntimeError(
            f"G4 FAIL: forward not wired — cannot satisfy required args {missing}. "
            f"signature={list(params)}; filled={list(kwargs)}"
        )
    if "hidden_states" not in kwargs and required:
        # Some models use different primary name
        if not kwargs:
            raise RuntimeError(
                "G4 FAIL: forward not wired — empty kwargs for "
                f"signature={list(params)}"
            )
    return kwargs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit", type=Path, required=True, help="Comfy int8 DiT safetensors")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Hybrid model dir (expects transformer/config.json)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit transformer config.json (overrides model-dir)",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--require-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), exit 1 unless forward runs",
    )
    args = p.parse_args(argv)

    if not args.dit.is_file():
        print(f"missing dit: {args.dit}", file=sys.stderr)
        return 2

    from h3_infer.comfy_dit_g0 import classify_dit_file
    from h3_infer.comfy_dit_load import load_comfy_int8_dit
    from h3_infer.meta_init import empty_minimax_h3_transformer
    from h3_infer.minimax_h3_convert import SourceLayout

    g0 = classify_dit_file(args.dit)
    print(
        json.dumps(
            {
                "g0": g0.verdict,
                "compatible": g0.compatible_with_stock_diffusers,
                "reasons": list(g0.reasons),
            }
        )
    )
    if not g0.compatible_with_stock_diffusers:
        print("G0 FAIL", g0.verdict)
        return 1

    config_path = args.config
    if config_path is None:
        if args.model_dir is None:
            print(
                "G4 FAIL: need --config or --model-dir with transformer/config.json",
                file=sys.stderr,
            )
            return 1
        config_path = Path(args.model_dir) / "transformer" / "config.json"
    if not config_path.is_file():
        print(
            f"G4 FAIL: missing local config {config_path} "
            "(run download_weights.py --pack hybrid_spike)",
            file=sys.stderr,
        )
        return 1

    device = torch.device(args.device)
    try:
        transformer = empty_minimax_h3_transformer(config_path)
    except Exception as exc:
        print(f"G4 FAIL: meta-init: {exc}")
        traceback.print_exc()
        return 1

    try:
        # materialize_nonpersistent_buffers runs INSIDE load — do not call again here
        info = load_comfy_int8_dit(
            transformer,
            args.dit,
            device=device,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
        print("load_ok", json.dumps({k: info[k] for k in info if k != "g1"}, default=str))
        print("g1", json.dumps(info["g1"], default=str))
    except Exception as exc:
        print(f"G1/G2/G3 FAIL: {exc}")
        traceback.print_exc()
        return 1

    if not args.require_forward:
        print("G4 FAIL: --no-require-forward is forbidden for PASS; exit 1")
        return 1

    try:
        dummy = _build_dummy_kwargs(transformer, device)
        with torch.no_grad():
            out = transformer(**dummy)
        tensors = _collect_tensors(out)
        if not tensors or not all(torch.isfinite(t).all() for t in tensors):
            print("G4 FAIL: empty or non-finite")
            return 1
    except Exception as exc:
        print(f"G4 FAIL: {exc}")
        traceback.print_exc()
        return 1

    print("G4 PASS", {"num_tensors": len(tensors), "shapes": [list(t.shape) for t in tensors]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
