# Krea 2 Multi-Ref Edit (N≤2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `image_edit` to accept 1–2 reference images on the existing dual-conditioning path (multi-image TE + multi source VAE tokens), without a new job type.

**Architecture:** Same identity-edit pipeline. `images[0]` (or explicit WH) sets canvas; each ref is fit independently; tokens cat with RoPE frames 1..N; target frame 0. Response stays top-level with additive `num_refs`.

**Tech Stack:** Python, PIL, PyTorch, existing `workers/krea2` unit tests (unittest).

**Spec:** `docs/superpowers/specs/2026-07-28-krea2-multi-ref-edit-design.md`

---

### Task 1: Request contract 1..2 images

**Files:**
- Modify: `workers/krea2/krea2_infer/request.py`
- Modify: `workers/krea2/tests/test_request.py`

- [ ] Update tests: edit accepts 2 images; rejects 0 and 3; size_from_source still from raw keys; two different sizes ok
- [ ] Change validation from exactly 1 to `1 <= len <= 2`
- [ ] Run: `PYTHONPATH=workers/krea2 python -m unittest workers.krea2.tests.test_request -v`

### Task 2: Encoder multi-image template + grounded_encode

**Files:**
- Modify: `workers/krea2/krea2_infer/encoder.py`
- Modify: `workers/krea2/tests/test_encoder_grounded.py`
- Create/Modify: `workers/krea2/tests/test_encoder_grounded_encode.py` (fake processor if needed)

- [ ] `grounded_template(instruction, num_images=1)` with 1..2 vision slots
- [ ] `grounded_encode(self, text: str, images: Sequence, *, grounding_px)` 
- [ ] Fake mm_processor + fake qwen unit tests (0/3 reject, 1/2 order, pad count)
- [ ] Run encoder tests

### Task 3: sample_edit multi-source + size xor

**Files:**
- Modify: `workers/krea2/krea2_infer/edit_sampling.py`
- Modify: `workers/krea2/tests/test_edit_sampling.py`

- [ ] `sources: Sequence[Image]` 1..2; xor WH reject
- [ ] Multi VAE encode + cat; multi pos/bias
- [ ] Tests for frames, boosts, xor size

### Task 4: Pipeline + handler

**Files:**
- Modify: `workers/krea2/krea2_infer/pipeline.py`
- Modify: `workers/krea2/handler.py`
- Optional: small pipeline unit test if importable lightly

- [ ] `edit(sources=..., source= deprecated)`
- [ ] Handler: `sources=norm.images`, top-level `num_refs`
- [ ] OOM message mention multi-ref tokens

### Task 5: README

**Files:**
- Modify: `workers/krea2/README.md`

- [ ] Document 1..2 images, size policy, scene-first, num_refs

### Task 6: Full test suite + commits

- [ ] `PYTHONPATH=workers/krea2 python -m unittest discover -s workers/krea2/tests -p 'test_*.py' -v`
- [ ] Commit logical slices
