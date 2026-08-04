# MiniMax H3 Level-1 Comfy DiT Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or disprove thin-Python load of Comfy-Org **non-pruned** FL2VA int8 ConvRot DiT into stock diffusers `MiniMaxH3Transformer3DModel` (no ComfyUI), with official TE — gates **G0–G4** (G5 optional). Pruned 19.5 GB file is **G0-only** (expected NO-GO).

**Architecture:** (1) G0 header classifies curve-AdaLN vs full time embed. (2) Convert with **explicit `source_layout`**: Comfy single-file = contiguous QKV **split only**; official raw shards = **reorder then split**. MLP half-swap + renames from pinned converter; int8 sides follow the same row ops. (3) **True meta-init** (`init_empty_weights` / `device="meta"`), then `assign=True` load, materialize non-persistent buffers (e.g. `rope.inv_freq`), then `.to(device)`. Patch Linear forward. (4) Real G4 forward or non-zero exit.

**Tech Stack:** Python 3.12, torch, safetensors, huggingface_hub, diffusers pin `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`, unittest.

**Spec:** `docs/superpowers/specs/2026-08-04-minimax-h3-level1-comfy-dit-spike-design.md` (rev: G0, converter-based map, assign+patch, hard G3/G4).

**Canonical references (do not reinvent):**

| Source | Use |
|--------|-----|
| `convert_minimax_h3_to_diffusers.py` L154–L273 + L757–L773 | Diffusers target layout; **reorder only on raw official shards** before `convert_transformer_key` |
| `comfy/ldm/minimax/model.py` L151–L158 (`qkv_proj` → `.split` thirds) | Comfy fused weight is already **contiguous** `[q_all;k_all;v_all]` — **no** `reorder_interleaved_qkv` |
| same model.py (`use_adaln_curves`, `adaln_t_table`) | Why pruned ≠ stock transformer |
| `workers/krea2/krea2_infer/pipeline.py` `_load_state_dict_into_dit` (`assign=True`, `requires_grad_(False)`) | Int8 Parameter load |
| `workers/krea2/krea2_infer/lora_runtime.py` `_patch_linear` | Runtime int8 forward |

---

## File map

| Path | Responsibility |
|------|----------------|
| `workers/minimax_h3/h3_infer/convrot.py` | From krea2 |
| `workers/minimax_h3/h3_infer/int8_linear.py` | From krea2 |
| `workers/minimax_h3/h3_infer/int8_linear_patch.py` | Minimal Linear.forward patch (krea2 lora_runtime subset, no LoRA) |
| `workers/minimax_h3/h3_infer/minimax_h3_convert.py` | Vendored/adapted converter key plan + transforms (bf16 + int8 sides) |
| `workers/minimax_h3/h3_infer/comfy_dit_g0.py` | G0 header classification (curve vs full) |
| `workers/minimax_h3/h3_infer/comfy_dit_load.py` | convert → assign load → patch → reports G1–G3 |
| `workers/minimax_h3/h3_infer/meta_init.py` | `init_empty_weights` / meta construct + `rope.inv_freq` materialize |
| `workers/minimax_h3/tools/spike_inspect_comfy_dit.py` | G0 (+ optional G1 dry) on real file |
| `workers/minimax_h3/tools/spike_dit_forward.py` | G4 real forward; fail-closed |
| `workers/minimax_h3/download_weights.py` | `--pack hybrid_spike` → **non-pruned** DiT default |
| `workers/minimax_h3/tests/test_*.py` | Unit tests per task |
| `docs/superpowers/specs/2026-08-04-minimax-h3-level1-comfy-dit-spike-report.md` | After real-file run |

**Default DiT artifact for hybrid download / G1–G4:**

`Comfy-Org/MiniMax-H3` → `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors`

**Inspect-only (G0 expected fail):**

`diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`

---

### Task 0: G0 — classify pruned/curve vs full AdaLN

**Files:**
- Create: `workers/minimax_h3/h3_infer/comfy_dit_g0.py`
- Create: `workers/minimax_h3/tests/test_comfy_dit_g0.py`
- Create: `workers/minimax_h3/tools/spike_inspect_comfy_dit.py`
- Create: `workers/minimax_h3/tools/__init__.py`

- [ ] **Step 1: Failing tests for G0 heuristics**

```python
# workers/minimax_h3/tests/test_comfy_dit_g0.py
from __future__ import annotations

import unittest

from h3_infer.comfy_dit_g0 import classify_dit_checkpoint, G0Result


class TestG0(unittest.TestCase):
    def test_curve_table_is_nogo(self):
        header = {
            "adaln_t_table": {"dtype": "F32", "shape": [1025, 8]},
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 8]},
        }
        r = classify_dit_checkpoint(header)
        self.assertEqual(r.verdict, "NO_GO_CURVE_ADALN")
        self.assertFalse(r.compatible_with_stock_diffusers)

    def test_full_time_embed_ok(self):
        # time_embed_dim=2688; adaln out = 6 * 3 * hidden = 6*3*5376 = 96768, in = 2688
        header = {
            "time_embedder.proj_in.weight": {"dtype": "F32", "shape": [5376, 256]},
            "time_embedder.proj_out.weight": {"dtype": "F32", "shape": [2688, 5376]},
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 2688]},
            "blocks.0.attn.qkv_proj.weight": {"dtype": "I8", "shape": [21504, 5376]},
        }
        r = classify_dit_checkpoint(header)
        self.assertEqual(r.verdict, "OK_FULL_ADALN")
        self.assertTrue(r.compatible_with_stock_diffusers)

    def test_narrow_adaln_without_table_still_nogo(self):
        header = {
            "blocks.0.adaln_proj.linear.weight": {"dtype": "I8", "shape": [96768, 8]},
        }
        r = classify_dit_checkpoint(header)
        self.assertFalse(r.compatible_with_stock_diffusers)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Implement `comfy_dit_g0.py`**

```python
# workers/minimax_h3/h3_infer/comfy_dit_g0.py
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


def classify_dit_checkpoint(header: dict[str, Any]) -> G0Result:
    """``header`` maps key -> {dtype, shape, ...} (safetensors header values)."""
    reasons: list[str] = []
    notes: list[str] = []
    keys = set(header)

    def shape_of(key: str) -> list[int] | None:
        if key not in header:
            return None
        return list(header[key]["shape"])

    # Curve form (Comfy pruned): shared time basis table + narrow AdaLN in_features.
    for k in keys:
        if k == "adaln_t_table" or k.endswith(".adaln_t_table") or k.endswith("adaln_t_table"):
            sh = shape_of(k)
            reasons.append(f"curve buffer {k} shape={sh} (stock diffusers has no adaln_t_table)")
            return G0Result(
                verdict="NO_GO_CURVE_ADALN",
                compatible_with_stock_diffusers=False,
                reasons=tuple(reasons),
            )

    # Any AdaLN linear with in_features != 2688 is incompatible with stock module.
    for k, meta in header.items():
        if "adaln_proj" in k and k.endswith(".weight"):
            sh = list(meta["shape"])
            if len(sh) >= 2 and sh[-1] != STOCK_TIME_EMBED_DIM:
                reasons.append(
                    f"{k} in_features={sh[-1]} != stock time_embed_dim={STOCK_TIME_EMBED_DIM}"
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
    if te_out is not None and te_out[0] != STOCK_TIME_EMBED_DIM:
        reasons.append(f"time_embedder out dim {te_out[0]} != {STOCK_TIME_EMBED_DIM}")
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


def classify_dit_file(path: str | Path) -> G0Result:
    return classify_dit_checkpoint(read_safetensors_header(path))
```

- [ ] **Step 3: CLI inspect — G0 only by default; exit 1 on NO_GO**

```python
#!/usr/bin/env python3
# tools/spike_inspect_comfy_dit.py
"""G0 inspect. Exit 0 only if compatible_with_stock_diffusers."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from h3_infer.comfy_dit_g0 import classify_dit_file, read_safetensors_header

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    args = p.parse_args(argv)
    if not args.path.is_file():
        print(f"missing: {args.path}", file=sys.stderr)
        return 2
    r = classify_dit_file(args.path)
    print(json.dumps({
        "path": str(args.path),
        "verdict": r.verdict,
        "compatible_with_stock_diffusers": r.compatible_with_stock_diffusers,
        "reasons": list(r.reasons),
        "notes": list(r.notes),
        "num_keys": len(read_safetensors_header(args.path)),
    }, indent=2))
    return 0 if r.compatible_with_stock_diffusers else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run unit tests**

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest tests.test_comfy_dit_g0 -v
```

Expected: OK.

- [ ] **Step 5: Document operator path**

On machine with weights:

```bash
# Expected: exit 1 NO_GO_CURVE_ADALN (or NARROW)
PYTHONPATH=. python tools/spike_inspect_comfy_dit.py \
  /path/to/minimax_h3_fl2va_pruned_int8_convrot.safetensors

# Expected: exit 0 OK_FULL_ADALN (if non-pruned matches theory)
PYTHONPATH=. python tools/spike_inspect_comfy_dit.py \
  /path/to/minimax_h3_fl2va_int8_convrot.safetensors
```

If non-pruned also fails G0 → **stop Level-1 inject path**; report NO-GO; do not invent curve port in this plan.

- [ ] **Step 6: Commit**

```bash
git add workers/minimax_h3/h3_infer/comfy_dit_g0.py \
  workers/minimax_h3/tests/test_comfy_dit_g0.py \
  workers/minimax_h3/tools/spike_inspect_comfy_dit.py \
  workers/minimax_h3/tools/__init__.py
git commit -m "feat(minimax_h3): G0 curve-AdaLN vs full checkpoint classifier"
```

---

### Task 1: Vendored converter transforms (explicit `source_layout` + int8 sides)

**Files:**
- Create: `workers/minimax_h3/h3_infer/minimax_h3_convert.py`
- Create: `workers/minimax_h3/tests/test_minimax_h3_convert.py`

**Do not** implement prefix-strip-only remap. **Copy** from pinned converter:

- `reorder_interleaved_qkv`, `split_fused_qkv`, `convert_transformer_key`
- `get_transformer_key_plan`, `MINIMAX_H3_TRANSFORMER_CONFIG`, **`MINIMAX_H3_TEST_TRANSFORMER_CONFIG`**
- `MINIMAX_H3_TRANSFORMER_DROPPED_KEYS`

#### Explicit QKV source layout (do not guess)

```python
class SourceLayout(str, Enum):
    """How fused ``*.attn.qkv_proj.weight`` rows are stored on disk."""

    # Official MiniMax shards: per-head interleaved [h0:qkv, h1:qkv, ...].
    # Upstream convert_transformer applies reorder_interleaved_qkv BEFORE convert_transformer_key
    # (diffusers script ~L757-L773).
    OFFICIAL_RAW_INTERLEAVED = "official_raw_interleaved"

    # Comfy-Org single-file DiT: weight is already contiguous [q_all; k_all; v_all].
    # Comfy Attention.forward does qkv_proj(x).split(heads*head_dim) — no load-time de-interleave
    # (comfy/ldm/minimax/model.py ~L151-L158). Level-1 hybrid MUST use this for Comfy files.
    COMFY_QKV_CONTIGUOUS = "comfy_qkv_contiguous"
```

| Layout | Before `split_fused_qkv` | Used for |
|--------|--------------------------|----------|
| `OFFICIAL_RAW_INTERLEAVED` | **`reorder_interleaved_qkv`** | MiniMaxAI `transformer/*.safetensors` shards |
| `COMFY_QKV_CONTIGUOUS` | **identity** (split only) | `Comfy-Org/.../*_int8_convrot.safetensors` |

`convert_transformer_key` itself always expects **reference / contiguous** QKV (see converter docstring). Layout only controls the **pre-step**.

Default for Level-1 hybrid load path: **`COMFY_QKV_CONTIGUOUS`**.  
Passing Comfy file with `OFFICIAL_RAW_INTERLEAVED` is a bug (double/wrong permute).

- [ ] **Step 1: Failing tests — use TEST config only (tiny tensors)**

```python
# workers/minimax_h3/tests/test_minimax_h3_convert.py
from __future__ import annotations

import unittest

import torch

from h3_infer.minimax_h3_convert import (
    MINIMAX_H3_TEST_TRANSFORMER_CONFIG as CFG,
    SourceLayout,
    convert_transformer_key,
    convert_transformer_key_with_sides,
    g1_int8_source_coverage,
    reorder_interleaved_qkv,
    split_fused_qkv,
)


class TestConvert(unittest.TestCase):
    def test_comfy_layout_splits_without_reorder(self):
        """Input is already [q_all;k_all;v_all]; layout=COMFY must match plain split."""
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        # Distinct blocks so wrong reorder is detectable
        q = torch.arange(heads * hd * hidden, dtype=torch.float32).reshape(heads * hd, hidden)
        k = q + 1000
        v = q + 2000
        w = torch.cat([q, k, v], dim=0)
        outs = convert_transformer_key_with_sides(
            {"blocks.0.attn.qkv_proj.weight": w},
            CFG,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
        self.assertTrue(torch.equal(outs["transformer_blocks.0.attn.to_q.weight"], q))
        self.assertTrue(torch.equal(outs["transformer_blocks.0.attn.to_k.weight"], k))
        self.assertTrue(torch.equal(outs["transformer_blocks.0.attn.to_v.weight"], v))

    def test_official_layout_reorders_then_splits(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        # Build interleaved raw, then expect contiguous thirds after convert
        # (use reorder_interleaved_qkv inverse construction or apply reorder as oracle)
        inner = heads * hd
        q = torch.randn(inner, hidden)
        k = torch.randn(inner, hidden)
        v = torch.randn(inner, hidden)
        contiguous = torch.cat([q, k, v], dim=0)
        # inverse of reorder: pack per-head qkv — implement helper or:
        # raw = interleave from (q,k,v); then convert with OFFICIAL must equal split(contiguous)
        raw = _interleave_qkv(q, k, v, heads, hd)  # implement in test module
        outs = convert_transformer_key_with_sides(
            {"blocks.0.attn.qkv_proj.weight": raw},
            CFG,
            source_layout=SourceLayout.OFFICIAL_RAW_INTERLEAVED,
        )
        oq, ok, ov = split_fused_qkv(contiguous, heads, hd)
        # allow float noise: use same tensors via reorder path
        reordered = reorder_interleaved_qkv(raw, heads, hd)
        tq, tk, tv = split_fused_qkv(reordered, heads, hd)
        self.assertTrue(torch.equal(outs["transformer_blocks.0.attn.to_q.weight"], tq))

    def test_mlp_fc1_half_swap(self):
        ffn, hidden = CFG["ffn_dim"], CFG["hidden_size"]
        gate = torch.ones(ffn, hidden)
        value = torch.full((ffn, hidden), 2.0)
        w = torch.cat([gate, value], dim=0)
        outs = convert_transformer_key("blocks.0.mlp.fc1.weight", w, CFG)
        self.assertEqual(outs[0][0], "transformer_blocks.0.ff.net.0.proj.weight")
        out = outs[0][1]
        self.assertTrue(torch.equal(out[:ffn], value))
        self.assertTrue(torch.equal(out[ffn:], gate))

    def test_int8_qkv_scale_splits_comfy_layout(self):
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        w = torch.randint(-8, 8, (rows, hidden), dtype=torch.int8)
        scale = torch.arange(rows, dtype=torch.float32)
        marker = torch.tensor(list(b'{"convrot":true,"convrot_groupsize":4}'), dtype=torch.uint8)
        state = {
            "blocks.0.attn.qkv_proj.weight": w,
            "blocks.0.attn.qkv_proj.weight_scale": scale,
            "blocks.0.attn.qkv_proj.comfy_quant": marker,
        }
        converted = convert_transformer_key_with_sides(
            state, CFG, source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS
        )
        third = heads * hd
        self.assertTrue(
            torch.equal(
                converted["transformer_blocks.0.attn.to_q.weight_scale"],
                scale[:third],
            )
        )
        self.assertIn("transformer_blocks.0.attn.to_q.comfy_quant", converted)

    def test_int8_mlp_fc1_scale_half_swap(self):
        ffn, hidden = CFG["ffn_dim"], CFG["hidden_size"]
        w = torch.randint(-4, 4, (2 * ffn, hidden), dtype=torch.int8)
        scale = torch.cat([torch.ones(ffn), torch.full((ffn,), 3.0)])
        converted = convert_transformer_key_with_sides(
            {"blocks.0.mlp.fc1.weight": w, "blocks.0.mlp.fc1.weight_scale": scale},
            CFG,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
        key = "transformer_blocks.0.ff.net.0.proj.weight_scale"
        self.assertTrue(torch.equal(converted[key][:ffn], torch.full((ffn,), 3.0)))
        self.assertTrue(torch.equal(converted[key][ffn:], torch.ones(ffn)))

    def test_g1_unknown_int8_weight_lowers_coverage(self):
        """Denominator = all int8 .weight in state, not planned∩present."""
        heads, hd = CFG["num_attention_heads"], CFG["attention_head_dim"]
        hidden = CFG["hidden_size"]
        rows = 3 * heads * hd
        good = torch.randint(-2, 2, (rows, hidden), dtype=torch.int8)
        orphan = torch.randint(-2, 2, (3, hidden), dtype=torch.int8)
        state = {
            "blocks.0.attn.qkv_proj.weight": good,
            "blocks.0.attn.qkv_proj.weight_scale": torch.ones(rows),
            "totally_unknown.int8.weight": orphan,  # not in plan
        }
        rep = g1_int8_source_coverage(
            state, CFG, source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS
        )
        self.assertEqual(rep["int8_weight_total"], 2)
        self.assertEqual(rep["int8_weight_mapped_ok"], 1)
        self.assertAlmostEqual(rep["source_coverage_ratio"], 0.5)
        self.assertTrue(any("totally_unknown" in u for u in rep["unmapped_int8_weights"]))


if __name__ == "__main__":
    unittest.main()
```

Implement `_interleave_qkv` in the test file (inverse of `reorder_interleaved_qkv`) for the official-layout test only.

- [ ] **Step 2: Implement `minimax_h3_convert.py`**

```python
def convert_transformer_key_with_sides(
    state: dict[str, torch.Tensor],
    config: dict,
    *,
    source_layout: SourceLayout,
) -> dict[str, torch.Tensor]:
    """Convert weights + side tensors. QKV pre-step depends on source_layout."""
    ...
    # For each *.attn.qkv_proj.weight:
    #   w = state[key]
    #   if source_layout is OFFICIAL_RAW_INTERLEAVED:
    #       w = reorder_interleaved_qkv(w, heads, head_dim)
    #       # same reorder on weight_scale if scale.ndim matches out-rows
    #   elif source_layout is COMFY_QKV_CONTIGUOUS:
    #       pass  # already [q_all;k_all;v_all]
    #   else: raise
    #   then split_fused_qkv / convert_transformer_key path; fan-out marker
    # For mlp.fc1: half-swap weight + scale (layout-independent)
    # Never call reorder when layout is COMFY_QKV_CONTIGUOUS
```

**Side-tensor rules:**

| Layout + weight op | `weight_scale` | `comfy_quant` |
|--------------------|----------------|---------------|
| COMFY: split thirds only | split dim0 into thirds | clone marker → to_q/to_k/to_v |
| OFFICIAL: reorder then split | reorder rows then split | clone marker |
| mlp.fc1 half-swap | half-swap dim0 | rename only |
| 1:1 rename | rename only | rename only |
| dropped keys | drop | drop |

- [ ] **Step 3: G1 metric — actual int8 weights, not planned∩present**

```python
def g1_int8_source_coverage(
    state: dict[str, torch.Tensor],
    config: dict,
    *,
    source_layout: SourceLayout,
) -> dict:
    """G1 primary ratio.

    denominator = count of tensors with dtype==int8 and key.endswith('.weight')
                  (after Comfy prefix strip on keys)
    numerator   = how many of those convert to ≥1 target with valid planned shape
                  under source_layout (no exception, shapes match plan)

    Unmapped / unknown keys stay in denominator and are listed in unmapped_int8_weights.
    """
    int8_weights = [
        k for k, t in state.items()
        if k.endswith(".weight") and t.dtype == torch.int8
    ]
    mapped_ok = []
    unmapped = []
    shape_errors = []
    for k in int8_weights:
        try:
            # convert single-key mini-state (include scale if present)
            ...
            if all shapes match plan targets:
                mapped_ok.append(k)
            else:
                shape_errors.append(...)
                unmapped.append(k)
        except Exception:
            unmapped.append(k)
    n = len(int8_weights)
    return {
        "int8_weight_total": n,
        "int8_weight_mapped_ok": len(mapped_ok),
        "source_coverage_ratio": len(mapped_ok) / max(n, 1),
        "unmapped_int8_weights": unmapped,
        "shape_errors": shape_errors,
    }
```

**Forbidden G1 definitions:**

- `mapped / len(module.state_dict())`
- `mapped / count(planned keys that appear in state)`  ← tautology if unknown keys ignored

Gate pass: `source_coverage_ratio >= 0.8` **and** `shape_errors == []` for mapped set (or fail any shape_error hard).

Comfy prefixes (`diffusion_model.`, …): strip **before** convert; converter still expects original MiniMax names (`blocks.*`, …).

- [ ] **Step 4: Run tests**

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest tests.test_minimax_h3_convert -v
```

Expected: OK; peak RAM small (TEST config only).

- [ ] **Step 5: Commit**

```bash
git add workers/minimax_h3/h3_infer/minimax_h3_convert.py workers/minimax_h3/tests/test_minimax_h3_convert.py
git commit -m "feat(minimax_h3): H3 convert with source_layout + real G1 int8 coverage"
```

---

### Task 2: Port int8 helpers + Linear patch + assign load

**Files:**
- Create: `workers/minimax_h3/h3_infer/convrot.py`
- Create: `workers/minimax_h3/h3_infer/int8_linear.py`
- Create: `workers/minimax_h3/h3_infer/int8_linear_patch.py`
- Create: `workers/minimax_h3/tests/test_int8_assign_and_patch.py`

- [ ] **Step 1: Failing test — assign keeps int8; patch matches dequant**

```python
# workers/minimax_h3/tests/test_int8_assign_and_patch.py
from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_infer.int8_linear import (
    apply_int8_side_tensors,
    int8_linear_forward,
    is_int8_linear,
    partition_int8_state_dict,
)
from h3_infer.int8_linear_patch import patch_module_int8_linears


class TestAssignAndPatch(unittest.TestCase):
    def test_assign_true_keeps_int8_dtype(self):
        m = nn.Linear(8, 4, bias=False)
        m.requires_grad_(False)
        w = torch.randint(-7, 7, (4, 8), dtype=torch.int8)
        scale = torch.ones(4)
        state = {"weight": w, "weight_scale": scale}
        weights, side = partition_int8_state_dict(
            {"proj.weight": w, "proj.weight_scale": scale}
        )
        # module path
        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4, bias=False)

        wrap = Wrap()
        wrap.requires_grad_(False)
        incompatible = wrap.load_state_dict(
            {"proj.weight": w}, strict=False, assign=True
        )
        self.assertEqual(len(incompatible.missing_keys), 0)
        self.assertEqual(wrap.proj.weight.dtype, torch.int8)
        n = apply_int8_side_tensors(wrap, side, {"proj.weight": w})
        self.assertEqual(n, 1)
        self.assertTrue(is_int8_linear(wrap.proj))

    def test_patch_matches_dequant_reference(self):
        wrap = nn.Linear(8, 4, bias=False)
        wrap.requires_grad_(False)
        w = torch.randint(-5, 5, (4, 8), dtype=torch.int8)
        scale = torch.linspace(0.01, 0.04, 4)
        wrap.load_state_dict({"weight": w}, strict=False, assign=True)
        apply_int8_side_tensors(
            nn.Module(),  # see below — use small Module with .proj
            {},
            {},
        )
        # Prefer a tiny container:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 4, bias=False)

        m = M()
        m.requires_grad_(False)
        m.load_state_dict({"proj.weight": w}, strict=True, assign=True)
        apply_int8_side_tensors(
            m,
            {"proj.weight_scale": scale},
            {"proj.weight": w},
        )
        patch_module_int8_linears(m, compute_dtype=torch.float32)
        x = torch.randn(2, 8)
        y = m.proj(x)
        # dequant reference
        w_f = w.float() * scale.float().unsqueeze(1)
        y_ref = F.linear(x.float(), w_f)
        self.assertTrue(torch.allclose(y.float(), y_ref, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
```

Clean up the test file when implementing (remove dead code in `test_patch_matches_dequant_reference` sketch).

- [ ] **Step 2: Copy krea2 `convrot.py` + `int8_linear.py`** with `h3_infer` imports.

- [ ] **Step 3: Implement patch**

```python
# workers/minimax_h3/h3_infer/int8_linear_patch.py
"""Patch nn.Linear.forward to use int8_linear_forward when weights are int8_tensorwise."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from h3_infer.int8_linear import int8_linear_forward, is_int8_linear

_PATCHED = "_h3_int8_patched"
_DTYPE = "_h3_int8_compute_dtype"


def patch_linear(module: nn.Linear, compute_dtype: torch.dtype) -> None:
    if hasattr(module, _PATCHED):
        setattr(module, _DTYPE, compute_dtype)
        return
    setattr(module, _PATCHED, True)
    setattr(module, _DTYPE, compute_dtype)

    def forward(x: torch.Tensor) -> torch.Tensor:
        dtype = getattr(module, _DTYPE)
        if is_int8_linear(module):
            return int8_linear_forward(x, module, dtype)
        return F.linear(x, module.weight, module.bias)

    module.forward = forward  # type: ignore[method-assign]


def patch_module_int8_linears(root: nn.Module, compute_dtype: torch.dtype = torch.bfloat16) -> int:
    n = 0
    for mod in root.modules():
        if isinstance(mod, nn.Linear) and is_int8_linear(mod):
            patch_linear(mod, compute_dtype)
            n += 1
    return n
```

- [ ] **Step 4: Run tests — pass**

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest tests.test_int8_assign_and_patch -v
```

- [ ] **Step 5: Commit**

```bash
git add workers/minimax_h3/h3_infer/convrot.py \
  workers/minimax_h3/h3_infer/int8_linear.py \
  workers/minimax_h3/h3_infer/int8_linear_patch.py \
  workers/minimax_h3/tests/test_int8_assign_and_patch.py
git commit -m "feat(minimax_h3): int8 assign load + Linear forward patch"
```

---

### Task 3: `comfy_dit_load` — G0 gate + convert + assign + patch + G1/G3 reports

**Files:**
- Create: `workers/minimax_h3/h3_infer/comfy_dit_load.py`
- Create: `workers/minimax_h3/tests/test_comfy_dit_load.py`

- [ ] **Step 1: Failing integration test on tiny synthetic “original layout” int8 state**

Use a **tiny** `nn.Module` that mimics **diffusers** names after convert (not Comfy names): e.g. only `transformer_blocks.0.attn.to_q` etc., and feed pre-original keys through `convert_transformer_key_with_sides`.

Also test: calling load without G0-compatible header raises / returns structured NO_GO.

- [ ] **Step 2: Implement loader**

```python
def load_comfy_int8_dit(
    module: nn.Module,
    path: Path,
    *,
    device: str | torch.device = "cpu",
    min_source_coverage: float = 0.8,
    compute_dtype: torch.dtype = torch.bfloat16,
    source_layout: SourceLayout = SourceLayout.COMFY_QKV_CONTIGUOUS,
) -> dict:
    # 1) G0 on header — if not compatible: raise RuntimeError with verdict (hard)
    # 2) load_file state
    # 3) strip known Comfy prefixes to original key names if needed
    # 4) g1_int8_source_coverage(state, config, source_layout=...) — fail if ratio < min
    #    or unmapped_int8_weights / shape_errors violate policy
    # 5) convert_transformer_key_with_sides(..., source_layout=source_layout)
    #    # NEVER default to official reorder for Comfy paths
    # 6) module.requires_grad_(False)  # module must already be meta- or empty-init by caller
    # 7) partition weights vs side; load_state_dict(weights, strict=False, assign=True)
    # 8) G3: missing required target keys → hard fail
    # 9) apply_int8_side_tensors; patch_module_int8_linears
    # 10) materialize_nonpersistent_buffers(module, device)  # SOLE call site for load path
    # 11) module.to(device) only after buffers exist
    # 12) return metrics
```

**Buffer materialize ownership:** only `load_comfy_int8_dit` (step 10) calls
`materialize_nonpersistent_buffers`. Callers (G4 CLI, tests) **must not** call it
again after a successful load. The helper itself is **idempotent** (see Task 5) so a
mistaken second call is a no-op, not a double-allocate hazard.

**G3 hard fail example:**

```python
if missing_required:
    raise RuntimeError(f"G3 FAIL missing {len(missing_required)} required keys e.g. {missing_required[:10]}")
```

**Forbidden:** `load_state_dict` without `assign=True` for int8.  
**Forbidden:** treating G3 as warning-only.  
**Forbidden:** `source_layout=OFFICIAL_RAW_INTERLEAVED` as default for Comfy-Org files.

- [ ] **Step 3: Unit tests pass**

- [ ] **Step 4: Commit**

```bash
git add workers/minimax_h3/h3_infer/comfy_dit_load.py workers/minimax_h3/tests/test_comfy_dit_load.py
git commit -m "feat(minimax_h3): Comfy/original int8 DiT load with hard G0-G3 gates"
```

---

### Task 4: Hybrid download (non-pruned default)

**Files:**
- Modify: `workers/minimax_h3/download_weights.py`
- Modify: `workers/minimax_h3/tests/test_download_weights.py`

- [ ] **Step 1: Tests**

```python
class TestHybridPack(unittest.TestCase):
    def test_hybrid_dry_run(self):
        self.assertEqual(main(["--output", "/tmp/x", "--pack", "hybrid_spike", "--dry-run"]), 0)

    def test_hybrid_downloads_non_pruned_dit(self):
        with mock.patch("download_weights._hf_hub_download", return_value="/tmp/dit") as hd:
            with mock.patch("download_weights._snapshot_download", return_value="/tmp/off") as sd:
                with mock.patch("pathlib.Path.mkdir"):
                    main(["--output", "/tmp/x", "--pack", "hybrid_spike"])
        self.assertIn("minimax_h3_fl2va_int8_convrot.safetensors", str(hd.call_args))
        self.assertNotIn("pruned", str(hd.call_args))
        patterns = sd.call_args.kwargs["allow_patterns"]
        joined = " ".join(patterns)
        self.assertIn("transformer/config.json", joined)
        # must not pull official ~66GB DiT weight shards
        self.assertNotIn("transformer/*.safetensors", joined)
        self.assertNotIn("transformer/**", joined)
```

- [ ] **Step 2: Implement**

```python
COMFY_REPO = "Comfy-Org/MiniMax-H3"
COMFY_DIT_NON_PRUNED = "diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors"
COMFY_DIT_PRUNED = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"  # G0 only

# MiniMaxAI snapshot for hybrid: TE/VAE/tokenizers + DiT *config only* (offline G4).
ALLOW_PATTERNS_HYBRID_TE_VAE = [
    "text_encoder/*",
    "text_encoder/**",
    "tokenizer/*",
    "tokenizer/**",
    "processor/*",
    "processor/**",
    "vae/*",
    "vae/**",
    "audio_vae/*",
    "audio_vae/**",
    "scheduler/*",
    "scheduler/**",
    "audio_scheduler/*",
    "audio_scheduler/**",
    "modular_model_index.json",
    "model_index.json",
    # Required for meta from_config / G4 without Hub network:
    "transformer/config.json",
    # Do NOT include transformer/*.safetensors or transformer/**
]

parser.add_argument("--pack", choices=("official", "hybrid_spike"), default="official")
parser.add_argument(
    "--also-fetch-pruned-for-g0",
    action="store_true",
    help="Also download pruned DiT for G0 inspect (not used for load)",
)
```

`hybrid_spike`:

1. hf_hub_download non-pruned DiT → `{output}/comfy/...`
2. optional pruned if flag
3. snapshot MiniMaxAI with `ALLOW_PATTERNS_HYBRID_TE_VAE` (TE/VAE/… + **`transformer/config.json` only**)

Print sizes: DiT ~34 GB + TE/VAE ~78 GB + config ≪1 MB ≈ 112 GB.

**G4 config path (offline):** `{MODEL_DIR}/transformer/config.json` from hybrid volume — not `load_config("MiniMaxAI/MiniMax-H3")` over the network after bootstrap.

- [ ] **Step 3: Tests pass + commit**

```bash
git commit -m "feat(minimax_h3): hybrid_spike downloads non-pruned Comfy int8 DiT"
```

---

### Task 5: G4 forward — fail closed

**Files:**
- Create: `workers/minimax_h3/tools/spike_dit_forward.py`

- [ ] **Step 1: Contract**

| Condition | Exit code |
|-----------|-----------|
| missing file / G0 NO_GO / G1–G3 fail | **1** |
| forward throws / non-finite | **1** |
| forward finite | **0** |
| “loaded OK” without calling forward | **FORBIDDEN** — treat as implementation bug; exit **1** if `--require-forward` default true |

- [ ] **Step 2: Meta-init helper (concrete — not a comment)**

`from_config(...)` without meta **allocates full bf16/float Parameters (~60GB+)**. Forbidden for G4 host path.

```python
# workers/minimax_h3/h3_infer/meta_init.py  (or inline in spike_dit_forward / comfy_dit_load)

def empty_minimax_h3_transformer(config_dict_or_path) -> MiniMaxH3Transformer3DModel:
    """Construct module on meta device (no full weight allocation)."""
    import torch
    from accelerate import init_empty_weights  # already in worker deps via accelerate
    from diffusers import MiniMaxH3Transformer3DModel

    cfg = MiniMaxH3Transformer3DModel.load_config(...)
    # Preferred:
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(cfg)
    # Fallback if accelerate unavailable:
    # with torch.device("meta"):
    #     model = MiniMaxH3Transformer3DModel.from_config(cfg)
    model.requires_grad_(False)
    return model


def materialize_nonpersistent_buffers(model: nn.Module, device: torch.device) -> None:
    """Idempotent: safe if called more than once; only ``load_comfy_int8_dit`` should call it.

    rope.inv_freq (and similar) are recomputed / non-persistent — not in Comfy weight files.

    After assign=True load of Parameters, any remaining meta buffers must be created
    before model.to(device) or first forward.

    Idempotence rules:
    - If buffer exists, is not on meta, and has correct shape/device → no-op for that buffer.
    - If buffer is missing or still meta → allocate/recompute once onto ``device``.
    - Never stack duplicate registers; use copy_ / existing register_buffer name.

    Implementation (match pin's MiniMaxH3RotaryPosEmbed):
    1. Module hook if exposed, e.g. model.rope.reset_inv_freq(), OR
    2. inv_freq = 1.0 / (theta ** (arange(0, 2*rope_freq_dim, 2) / (2*rope_freq_dim)))

    Must NOT rely on load_state_dict for rope.inv_freq (dropped by converter).
    """
    ...
```

**Load order (mandatory) — single materialize:**

1. `model = empty_minimax_h3_transformer(local_config_path)`  # meta; config from disk  
2. `load_comfy_int8_dit(model, path, device=..., source_layout=COMFY_…)`  
   - inside: `assign=True`  
   - inside: **`materialize_nonpersistent_buffers` once**  
   - inside: device move as needed  
3. forward (caller does **not** call materialize again)

If materialize is skipped inside load, `.to(cuda)` / forward often **crashes** on meta `rope.inv_freq`.

Unit test (CPU): (a) meta Linear + assign int8; (b) materialize twice on a stub buffer → same object/data, no error.

- [ ] **Step 3: Implementation outline for CLI**

```python
def main(argv=None) -> int:
    ...
    g0 = classify_dit_file(args.dit)
    if not g0.compatible_with_stock_diffusers:
        print("G0 FAIL", g0)
        return 1

    from h3_infer.meta_init import empty_minimax_h3_transformer
    from h3_infer.minimax_h3_convert import SourceLayout

    # Prefer local hybrid layout (no Hub):
    #   {model_dir}/transformer/config.json  from --pack hybrid_spike
    config_path = args.config or (Path(args.model_dir) / "transformer" / "config.json")
    if not config_path.is_file():
        print(f"G4 FAIL: missing local config {config_path} (run hybrid_spike download)")
        return 1

    transformer = empty_minimax_h3_transformer(config_path)  # reads file, not network
    try:
        # materialize_nonpersistent_buffers runs INSIDE load — do not call again here
        info = load_comfy_int8_dit(
            transformer,
            args.dit,
            device=args.device,
            source_layout=SourceLayout.COMFY_QKV_CONTIGUOUS,
        )
    except Exception as exc:
        print("G1/G2/G3 FAIL", exc)
        return 1

    # READ MiniMaxH3Transformer3DModel.forward in installed pin; construct legal dummy inputs.
    # If signature cannot be satisfied, exit 1 — do NOT exit 0.
    try:
        with torch.no_grad():
            out = transformer(**dummy_kwargs)
        tensors = _collect_tensors(out)
        if not tensors or not all(torch.isfinite(t).all() for t in tensors):
            print("G4 FAIL: empty or non-finite")
            return 1
    except Exception as exc:
        print("G4 FAIL", exc)
        return 1

    print("G4 PASS", info)
    return 0
```

Wire dummy inputs **only** after reading the real forward on the GPU host; until then exit **1** with `G4 FAIL: forward not wired` — **never 0**.

- [ ] **Step 4: Commit**

```bash
git add workers/minimax_h3/h3_infer/meta_init.py workers/minimax_h3/tools/spike_dit_forward.py
git commit -m "feat(minimax_h3): meta-init + fail-closed G4 DiT forward spike"
```

---

### Task 6: Spike report + README

**Files:**
- Create: `docs/superpowers/specs/2026-08-04-minimax-h3-level1-comfy-dit-spike-report.md`
- Modify: `workers/minimax_h3/README.md`

Template:

```markdown
# Level-1 Comfy DiT spike report

| Gate | File | Result | Evidence |
|------|------|--------|----------|
| G0 | pruned | | |
| G0 | non-pruned | | |
| G1 source coverage | non-pruned | | ratio=, shape_errors= |
| G2 assign+patch | | | dequant test / int8_layers= |
| G3 target completeness | | | missing= |
| G4 forward | | | finite= |
| G5 | | SKIP/… | |

**Verdict:** GO only if G0–G4 PASS on non-pruned full-AdaLN file.
**Pruned:** not a Level-1 GO path (curve AdaLN).
```

- [ ] **Step 1: Fill after real runs**
- [ ] **Step 2: Commit report**

---

## Self-review vs blocking defects

| Defect | Fix in this plan |
|--------|------------------|
| P1 pruned ≠ stock transformer | **Task 0 G0**; primary DiT = **non-pruned**; curve port out of scope |
| P1 prefix-strip ≠ conversion | **Task 1** renames + MLP swap + layout-aware QKV |
| P1 Comfy QKV double-reorder | **`SourceLayout.COMFY_QKV_CONTIGUOUS`**: split only; reorder only for `OFFICIAL_RAW_INTERLEAVED` |
| P1 load_state_dict casts int8→float | **Task 2** `requires_grad_(False)` + `assign=True` |
| P1 no int8 forward | **Task 2** `int8_linear_patch` + dequant numerical test |
| P1 full 33B alloc on from_config | **Task 5** `init_empty_weights` / meta + buffer materialize (`rope.inv_freq`) |
| G4 needs network for config | **Task 4** hybrid includes `transformer/config.json` only |
| Double materialize | **Sole call** inside `load_comfy_int8_dit`; helper **idempotent**; G4 CLI does not re-call |
| P1 G4 success without forward | **Task 5** fail-closed; exit 1 if no forward |
| P1 G3 soft | **Task 3** hard RuntimeError on missing required |
| P1/P2 G1 tautology | **Task 1** `g1_int8_source_coverage`: denom = all int8 `.weight`; unknown keys hurt ratio |
| P2 huge test tensors | **Task 1** tests use **`MINIMAX_H3_TEST_TRANSFORMER_CONFIG` only** |

---

## Execution notes

1. Run Tasks 0–3 on CPU CI/dev without full weights (synthetic + unit tests).  
2. Download hybrid non-pruned on volume; G0 pruned vs non-pruned with inspect CLI.  
3. G4 only after forward signature wired from installed diffusers.  
4. **Do not** change production `H3Pipeline` default in this plan.  
5. If non-pruned G0 fails → Level-1 inject is dead; report recommends torchao or Comfy headless.
