# MiniMax H3 Level-1 spike: Comfy DiT int8 without ComfyUI

**Дата:** 2026-08-04  
**Статус:** approved (design) — research spike, not production default  
**Связано:** `docs/superpowers/specs/2026-08-04-minimax-h3-t2v-worker-design.md` (MVP backend B)  
**Plan:** `docs/superpowers/plans/2026-08-04-minimax-h3-level1-comfy-dit-spike.md`

## Контекст

MVP worker (`workers/minimax_h3/`) loads **MiniMaxAI/MiniMax-H3** via diffusers
ModularPipeline (~144 GB t2va half, 1×80 GB + CPU offload).

Comfy-Org ships much smaller DiT checkpoints, e.g.
`minimax_h3_fl2va_pruned_int8_convrot.safetensors` (~19.5 GiB). Those files are
not a drop-in for `ModularPipeline.from_pretrained`.

**krea2** already proves a pattern: thin Python loads Comfy **int8_tensorwise +
ConvRot** DiT **without** ComfyUI (`krea2_infer/int8_linear.py`, `convrot.py`).

This spike asks: can we do the **same class of thing for MiniMax H3 DiT**, while
keeping **official text encoder** (hybrid), and stay off ComfyUI runtime?

## Goal

Prove or disprove **Level-1 hybrid load** on a thin Python path:

| Component | Source |
|-----------|--------|
| DiT (FL2VA / t2va) | **Primary for Level-1 load:** Comfy-Org **`minimax_h3_fl2va_int8_convrot.safetensors`** (~34 GB, **non-pruned**). **Pruned** `…_pruned_int8_convrot` (~19.5 GB) is **inspect-only → expected G0 NO-GO** for stock diffusers (see Prune / curve AdaLN). |
| Text encoder | MiniMaxAI `text_encoder/` (bf16 sharded; **not** nvfp4 AWQ) |
| Video VAE | MiniMaxAI `vae/` first; Comfy single-file only if keys match |
| Audio VAE | MiniMaxAI `audio_vae/` first; Comfy single-file only if keys match |
| Runtime | Python + pinned diffusers MiniMax-H3 — **no ComfyUI** |
| Key map | **Must reuse** pinned `convert_minimax_h3_to_diffusers.py` renames + MLP half-swap + QKV **split**. QKV **reorder** only when `source_layout=official_raw_interleaved` (sharded MiniMax). **Comfy single-file** = `source_layout=comfy_qkv_contiguous` → **split only** (Comfy `qkv_proj` forward already assumes contiguous `[q;k;v]` thirds). Extend transforms to int8 `weight_scale` / `comfy_quant`. |

Success is a documented **GO / NO-GO**, not a silent switch of production default.

## Non-goals

- Headless ComfyUI / workflow JSON execution.
- Loading `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (Level-2 TE).
- Ref2VA / `transformer_ref` / R2V.
- Changing public RunPod API or replacing MVP backend B as default.
- Bit-exact parity with Comfy 20-step templates.
- Production SLO, multi-GPU, or volume size optimization as a hard goal.

## Architecture

```text
                    ┌─────────────────────────────────────┐
  Comfy DiT file ──►│ comfy_dit_load (inspect + materialize)│
                    │  int8_tensorwise + ConvRot (+ prune) │
                    └──────────────┬──────────────────────┘
                                   │ state → module
                                   ▼
                    MiniMaxH3Transformer3DModel (diffusers)
                                   │
  MiniMaxAI TE  ──► Qwen3-VL conditioner (unchanged official load)
  MiniMaxAI VAE ──► video + audio VAE (official; Comfy optional)
                                   │
                                   ▼
                    ModularPipeline / blocks  OR  isolated DiT forward
```

**Production `handler.py` / default `H3Pipeline` stay on official pack** until this
spike returns GO and a follow-up design promotes hybrid to an opt-in mode
(e.g. `DIT_SOURCE=comfy_int8`).

### Reuse from krea2

| Piece | Role |
|-------|------|
| `partition_int8_state_dict` | split weights vs `.weight_scale` / `.comfy_quant` |
| `parse_comfy_quant_marker` | ConvRot flags / group size |
| `materialize_int8_linear` / `apply_int8_side_tensors` | bind int8 onto `nn.Linear` |
| `convrot` Hadamard helpers | online activation rotate |
| `int8_linear_forward` | W8A8 or dequant fallback |

Copy/adapt under `workers/minimax_h3/h3_infer/` for the spike (do **not** create a
shared package until GO — avoid premature coupling).

### Prune / curve AdaLN (blocking for stock diffusers)

Comfy **pruned** checkpoints are **not** “same modules, fewer bytes”. They change
the **time / AdaLN path** (see ComfyUI
`comfy/ldm/minimax/model.py`):

- `use_adaln_curves` when `adaln_curve_grid` is set;
- buffer `adaln_t_table` e.g. shape **`[1025, 8]`** (curve basis), **not** full
  `TimeEmbedder` → `time_embed_dim=2688`;
- AdaLN projection weights on a **narrow** t_dim (e.g. **8**), not 2688;
- forward **interpolates** the curve table instead of running `time_embedder`.

Stock `MiniMaxH3Transformer3DModel` (diffusers) expects the **full** time MLP +
AdaLN `in_features=time_embed_dim` (2688). **Prefix remap / `load_state_dict` cannot
recover this** without either:

- reimplementing curve AdaLN inside or around the diffusers module (out of
  Level-1 thin scope), or
- running Comfy’s model class.

**Level-1 rule:**

| File | Role |
|------|------|
| `…_pruned_int8_convrot` | **G0 header inspect only.** If `adaln_t_table` or AdaLN `in_features != 2688` → **hard NO-GO** for inject-into-diffusers. Do **not** proceed to G1–G4 on this file. |
| `…_int8_convrot` (non-pruned) | **Primary** Level-1 load target for G1–G4. |

Substring “modulation missing” heuristics are **insufficient** as the sole G0/G3
signal — shape/header of AdaLN / `adaln_t_table` is authoritative.

Curve-AdaLN port is **explicitly out of Level-1**; may be Level-1b only if
non-pruned GO already works and product still wants 19.5 GB DiT.

## Disk / hardware (spike)

| Component | ~size |
|-----------|------:|
| Comfy DiT **non-pruned** int8 (primary) | ~34 GB |
| Comfy DiT pruned int8 (inspect only) | ~20 GB |
| Official text_encoder | ~67 GB |
| Official vae + audio_vae | ~11 GB |
| **Total hybrid (non-pruned path)** | **~112 GB** |

- Volume: **≥ 130 GB** for hybrid non-pruned spike.
- GPU: **1×80 GB** class preferred for G4; TE offload as official recipe.

## Gates (ordered)

| ID | Gate | Pass criterion | Fail |
|----|------|----------------|------|
| **G0** | Checkpoint class | Header (or first tensors) prove **full** AdaLN path: no `adaln_t_table` curve checkpoint; AdaLN / time_embed shapes match diffusers `time_embed_dim=2688` plan. | **Hard NO-GO** for this file; switch to non-pruned or stop |
| **G1** | Source map + shapes | **Denominator** = every tensor in the checkpoint with `dtype=int8` and key suffix `.weight` (after optional Comfy prefix strip). **Numerator** = those that convert under the declared `source_layout` to ≥1 target key with **valid planned shape**. Unplanned / unknown int8 weights **count against** coverage (not ignored). Track separately: shape/transform errors; G3 = target completeness. **Not** planned∩present tautology; **not** matched/target-only. | Hard fail |
| **G2** | Int8 materialize + patch | **True meta-init** (`accelerate.init_empty_weights` or `torch.device("meta")`), not bare `from_config` on CPU; `requires_grad_(False)`; `load_state_dict(..., assign=True)` so weights stay **int8**; materialize non-persistent buffers (`rope.inv_freq`) before `.to(device)`/forward; side scales/markers; **Linear forward patch** → `int8_linear_forward`. Unit test: dequant numerical parity. | Hard fail |
| **G3** | Target completeness | After convert+load: no unexplained **missing** required target weights for a full (non-curve) DiT; prune/curve residues → fail. **Hard fail**, not warning-only. | Hard fail |
| **G4** | Forward | One real DiT forward (dummy packed/video+audio inputs per actual `forward` signature); **all outputs finite**. Exit code **non-zero** on failure. **Must not** return 0 without executing forward. | Hard fail |
| **G5** | Smoke (soft) | Optional short t2va; does not alone imply GO. | Soft |

**GO (promote hybrid to opt-in design):** G0–G4 pass on **non-pruned** Comfy int8 (or equivalent full-AdaLN file); G5 preferred.  
**NO-GO:** any hard fail G0–G4; write spike report; keep MVP on MiniMaxAI only.  
Pruned-only success is **not** a Level-1 GO.  
Alternatives after NO-GO: official + torchao int8 (docs), or headless Comfy (separate spec).

## Deliverables

1. **Spec** (this file) + **plan**.
2. `download_weights.py` **hybrid_spike** mode: default Comfy DiT =
   **non-pruned** `minimax_h3_fl2va_int8_convrot.safetensors` + MiniMaxAI TE/VAE
   + **`transformer/config.json` only** (no official DiT weight shards) so G4
   meta-init is offline; optional pruned DiT **for G0 inspect only**.
3. `h3_infer/` converter port of `convert_transformer_key` / QKV / MLP (from pinned
   diffusers script) + int8 side-tensor transforms; int8 materialize with
   **assign=True** + **Linear forward patch**; **single** idempotent
   `materialize_nonpersistent_buffers` inside the load path (not also in G4 CLI).
4. CPU unit tests: G0 header heuristics; convert plan shapes; int8 assign+patch
   numerical dequant parity; no “coverage = matched/target only” as sole G1.
5. CLI: G0 inspect; G4 forward that **fails closed** (no success without forward).
6. Spike report: GO/NO-GO with G0–G4 evidence.

## API / product

No client-visible change during spike. Internal only:

| Env / flag | Purpose |
|------------|---------|
| `DIT_SOURCE=official` (default) | Current ModularPipeline snapshot |
| `DIT_SOURCE=comfy_int8` | Only after GO; hybrid load (follow-up) |

Spike may use CLI tools without wiring env into handler.

## License

Same MiniMax H3 Community License. Comfy-Org pack is a repack of MiniMax works;
territorial and notice obligations still apply. Do not treat civitai mirrors as
canonical — prefer Hugging Face `Comfy-Org/MiniMax-H3` + `MiniMaxAI/MiniMax-H3`.

## Risks

| Risk | Mitigation |
|------|------------|
| Pruned = curve AdaLN ≠ stock transformer | **G0 hard NO-GO**; primary file is non-pruned int8 |
| Key layout ≠ diffusers | Reuse pinned `convert_minimax_h3_to_diffusers.py` transforms |
| int8 cast to float Parameter | meta-init + `requires_grad_(False)` + `assign=True` (krea2) |
| int8 weights never used in matmul | Patch Linear forward → `int8_linear_forward` + dequant test |
| False G4 success | Forward required; non-zero exit without it |
| False G1 on pruned | Source coverage + shape validation; not target-only ratio |
| TE still ~67 GB | Accepted for Level-1; nvfp4 is Level-2 |

## Success metrics

- Clear **GO** or **NO-GO** within one focused engineering pass.
- If GO: listed steps to opt-in hybrid mode + residual work (offload, steps, quality).
- If NO-GO: one-paragraph recommendation (torchao vs Comfy headless) with evidence
  from G1–G3.

## Out of order for later levels

| Level | Content |
|-------|---------|
| 2 | Comfy nvfp4 TE in thin Python |
| 3 | Full ~40 GB Comfy pack parity without ComfyUI |
| C/D | Headless Comfy worker (different backend) |
