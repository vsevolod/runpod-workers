"""G0: is this Comfy/original DiT loadable into stock MiniMaxH3Transformer3DModel?"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Match MINIMAX_H3_TRANSFORMER_CONFIG.time_embed_dim
STOCK_TIME_EMBED_DIM = 2688
STOCK_HIDDEN = 5376
STOCK_ADALN_OUT = 6 * 3 * STOCK_HIDDEN  # 96768


@dataclass(frozen=True)
class G0Result:
    verdict: str
    compatible_with_stock_diffusers: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...] = ()


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


def classify_dit_checkpoint(
    header: dict[str, Any],
    *,
    time_embed_dim: int = STOCK_TIME_EMBED_DIM,
) -> G0Result:
    """``header`` maps key -> {dtype, shape, ...} (safetensors header values).

    ``time_embed_dim`` defaults to stock MiniMax-H3 (2688). Tests may pass the
    TEST config value; production load uses stock / full config.
    """
    reasons: list[str] = []
    notes: list[str] = []
    keys = set(header)

    def shape_of(key: str) -> list[int] | None:
        if key not in header:
            return None
        return list(header[key]["shape"])

    # Curve form (Comfy pruned): shared time basis table + narrow AdaLN in_features.
    for k in keys:
        if k == "adaln_t_table" or k.endswith(".adaln_t_table") or k.endswith(
            "adaln_t_table"
        ):
            sh = shape_of(k)
            reasons.append(
                f"curve buffer {k} shape={sh} (stock diffusers has no adaln_t_table)"
            )
            return G0Result(
                verdict="NO_GO_CURVE_ADALN",
                compatible_with_stock_diffusers=False,
                reasons=tuple(reasons),
            )

    # Any AdaLN linear with in_features != expected time_embed_dim is incompatible.
    for k, meta in header.items():
        if "adaln_proj" in k and k.endswith(".weight"):
            sh = list(meta["shape"])
            if len(sh) >= 2 and sh[-1] != time_embed_dim:
                reasons.append(
                    f"{k} in_features={sh[-1]} != stock time_embed_dim={time_embed_dim}"
                )
                return G0Result(
                    verdict="NO_GO_NARROW_ADALN",
                    compatible_with_stock_diffusers=False,
                    reasons=tuple(reasons),
                )

    # Prefer positive evidence of full time embedder.
    te_out = shape_of("time_embedder.proj_out.weight") or shape_of(
        "time_embedder.linear_2.weight"
    )
    if te_out is not None and te_out[0] != time_embed_dim:
        reasons.append(f"time_embedder out dim {te_out[0]} != {time_embed_dim}")
        return G0Result(
            verdict="NO_GO_TIME_EMBED",
            compatible_with_stock_diffusers=False,
            reasons=tuple(reasons),
        )

    if te_out is None and not any("adaln_proj" in k for k in keys):
        notes.append("no time_embedder or adaln_proj in header sample; inconclusive")
        return G0Result(
            verdict="INCONCLUSIVE",
            compatible_with_stock_diffusers=False,
            reasons=("insufficient keys to classify",),
            notes=tuple(notes),
        )

    return G0Result(
        verdict="OK_FULL_ADALN",
        compatible_with_stock_diffusers=True,
        reasons=("no curve table; AdaLN/time shapes consistent with stock",),
        notes=tuple(notes),
    )


def classify_dit_file(
    path: str | Path, *, time_embed_dim: int = STOCK_TIME_EMBED_DIM
) -> G0Result:
    return classify_dit_checkpoint(
        read_safetensors_header(path), time_embed_dim=time_embed_dim
    )
