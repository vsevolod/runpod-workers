# Level-1 Comfy DiT spike report

**Date:** 2026-08-05  
**Branch:** master (implementation of plan `2026-08-04-minimax-h3-level1-comfy-dit-spike`)  
**Production default:** unchanged (official MiniMaxAI ModularPipeline)

## Gates

| Gate | File | Result | Evidence |
|------|------|--------|----------|
| G0 unit | synthetic headers | **PASS** | `tests/test_comfy_dit_g0.py` — curve → `NO_GO_CURVE_ADALN`; full AdaLN → `OK_FULL_ADALN`; narrow → `NO_GO_NARROW_ADALN` |
| G0 | pruned real | **PENDING** | needs volume download (`--also-fetch-pruned-for-g0`); expected **NO_GO** |
| G0 | non-pruned real | **PENDING** | needs `minimax_h3_fl2va_int8_convrot.safetensors`; expected **OK** if full AdaLN |
| G1 source coverage | unit | **PASS** (synthetic) | `tests/test_minimax_h3_convert.py` — unknown int8 lowers ratio; QKV/MLP side ops |
| G1 | non-pruned real | **PENDING** | run `load_comfy_int8_dit` / G4 CLI after download |
| G2 assign+patch | unit | **PASS** | `tests/test_int8_assign_and_patch.py` — `assign=True` keeps int8; patch matches dequant |
| G3 target completeness | unit | **PASS** (tiny module) | `tests/test_comfy_dit_load.py` — G0 hard fail; int8 QKV load on stub |
| G4 forward | real DiT | **PENDING** | `tools/spike_dit_forward.py` fail-closed; needs 80 GB-class host + hybrid pack (~112 GB). Local host (8 GB 3070 Ti, ~35 GB free disk) cannot complete full hybrid download or full DiT forward. |
| G5 | — | **SKIP** | out of spike scope for this session |

## Deliverables landed (code)

| Path | Role |
|------|------|
| `h3_infer/comfy_dit_g0.py` | G0 header classifier |
| `h3_infer/minimax_h3_convert.py` | `SourceLayout`, convert + int8 sides, G1 coverage |
| `h3_infer/convrot.py` / `int8_linear.py` | krea2 port |
| `h3_infer/int8_linear_patch.py` | Linear.forward patch |
| `h3_infer/meta_init.py` | meta construct + idempotent buffer materialize |
| `h3_infer/comfy_dit_load.py` | G0→G3 load; sole materialize call site |
| `download_weights.py --pack hybrid_spike` | non-pruned DiT + TE/VAE + `transformer/config.json` |
| `tools/spike_inspect_comfy_dit.py` | G0 CLI |
| `tools/spike_dit_forward.py` | G4 fail-closed CLI |

## Verdict

**INCOMPLETE / not GO yet** for promoting hybrid to opt-in product path.

Reasons:

1. Real Comfy DiT files were **not** run on this machine (disk ~35 GB free ≪ ~112 GB hybrid; no Hub cache).
2. Full G4 requires GPU host with enough VRAM/offload path for int8 DiT (~34 GB weights) plus wired forward signature against pinned diffusers.
3. Unit gates G0–G3 (synthetic) and download-pack tests are green; implementation matches plan hard-fail contracts.

**Next operator steps (80 GB pod + ≥150 GB volume):**

```bash
python download_weights.py --output $MODEL_DIR --pack hybrid_spike --also-fetch-pruned-for-g0
PYTHONPATH=. python tools/spike_inspect_comfy_dit.py $MODEL_DIR/comfy/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
# expect exit 1 NO_GO
PYTHONPATH=. python tools/spike_inspect_comfy_dit.py $MODEL_DIR/comfy/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
# expect exit 0 OK_FULL_ADALN — if not, Level-1 inject is NO-GO
PYTHONPATH=. python tools/spike_dit_forward.py \
  --dit $MODEL_DIR/comfy/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors \
  --model-dir $MODEL_DIR
# expect G4 PASS or hard fail with reason
```

Update this table with real exit codes and ratios, then set:

- **GO** only if G0–G4 PASS on non-pruned full-AdaLN file  
- **NO-GO** if any hard gate fails; keep MVP on MiniMaxAI only  

**Pruned:** not a Level-1 GO path (curve AdaLN).
