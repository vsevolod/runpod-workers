# Krea 2 LoRA type + weight_diff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional `loras[].type` (`lora` | `weight_diff`) with per-type strength ceilings and runtime application of full weight deltas (e.g. Fedor-style projector.diff).

**Architecture:** Extend request normalization with explicit type discriminator; branch `LoRALoader` by type; keep one Linear forward that sums low-rank and weight-diff deltas; files stay on volume allowlist.

**Tech Stack:** Python 3.12, PyTorch, safetensors, unittest

**Spec:** `docs/superpowers/specs/2026-07-26-krea2-lora-type-weight-diff-design.md`

---

## File map

| File | Responsibility |
|------|----------------|
| `workers/krea2/krea2_infer/lora.py` | type constants, selection, normalize ceilings, load lora vs weight_diff |
| `workers/krea2/krea2_infer/lora_runtime.py` | activate both layer kinds; forward math |
| `workers/krea2/tests/test_lora.py` | normalize + load contract tests |
| `workers/krea2/tests/test_lora_runtime.py` | weight_diff runtime math + mix + cleanup |
| `workers/krea2/README.md` | API docs |

### Task 1: Normalize `type` + per-type strength

**Files:** `lora.py`, `test_lora.py`

- [ ] Extend `LoRASelection` with `type`, `as_dict` includes type
- [ ] Default type `lora`; allow `weight_diff`; ceilings 2.0 / 5.0
- [ ] Update existing as_dict expectations; add new normalize tests
- [ ] Run tests; commit

### Task 2: Load `weight_diff`

**Files:** `lora.py`, `test_lora.py`

- [ ] `WeightDiffLayerWeights`; branch loader; reject type/content mismatch
- [ ] Map `diffusion_model.txtfusion.projector.diff` → `txtfusion.projector`
- [ ] Tests: happy path, wrong shape, A/B with weight_diff type, .diff with lora type
- [ ] Run tests; commit

### Task 3: Runtime weight_diff + mix

**Files:** `lora_runtime.py`, `test_lora_runtime.py`

- [ ] Active layer supports delta; forward adds `linear(x, delta)*strength`
- [ ] Tests: math vs fused weight; mix lora+diff; deactivate clears both
- [ ] Run full suite; commit

### Task 4: README

**Files:** `workers/krea2/README.md`

- [ ] Document type, ceilings, weight_diff example
- [ ] Commit
