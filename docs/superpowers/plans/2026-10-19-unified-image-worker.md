# Unified image worker implementation plan

> **For agentic workers:** Use the executing-plans workflow inline for this tightly coupled runtime migration. Follow test-first implementation and verify each checkpoint before deployment.

**Goal:** Serve Krea and Qwen through one extensible worker and retained endpoint.

**Architecture:** A registry routes validated requests to model backends. A
single manager serializes inference and owns the active backend; backend
closing frees model references before the next model loads. Existing
model-specific parsers, LoRA handling and inference implementations are reused.

**Tech Stack:** Python 3.12, RunPod, PyTorch, Transformers, Diffusers.

**Spec:** `docs/superpowers/specs/2026-10-19-unified-image-worker-design.md`.

## Global constraints

- Retain endpoint ID `9zb0wyo61ck3wk`; final name `image-workers`.
- Models: `krea2`, `qwen_image_21`; default `krea2`.
- Qwen Base/Turbo stays in `model_options.variant`.
- One active backend, complete inference serialized, previous model freed before loading the next.
- No checkpoint is selected, downloaded or loaded at startup.
- Do not mutate process environment per request.
- Keep each model's existing LoRA directory and request semantics.
- CUDA 12.8.1, Torch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 5.20.1.
- Diffusers commit da1d3829cf08d4f329b526d89e17cc035c049d8d and native-reference patch.
- Deploy and verify the exact published image digest.
- Stop old Qwen endpoint only after common-worker checks succeed and its queue is clear.

## Task 1: Backend ownership manager

Files: `workers/image/image_worker/manager.py`, `tests/test_manager.py`.

- [x] Write failing tests for lazy initialization, warm reuse, Krea/Qwen/Krea
  switching, releasing old backend/resource weak references before replacement
  loading, unknown models preserving current state, failed load cleanup/retry,
  and an in-flight call blocking a competing model switch.
- [x] Run the new suite and observe the missing implementation fail.
- [x] Implement `ModelManager(registry)` with `infer(model, norm, seed,
  raw_loras)` and idempotent `close()`. Registry lookup precedes unloading.
  Use one `RLock`, clear ownership before closing and collect references
  before CUDA cache cleanup. Log model load/unload and allocated memory.
- [x] Re-run the suite and verify every listed lifecycle case.

```sh
python -m unittest discover -s workers/image/tests -p test_manager.py
```

## Task 2: Model backends and resource release

Files: `image_worker/backends.py`, model-specific `api.py`, Qwen
`model_manager.py`, `runtime_config.py`, `pipeline.py`, and their tests.

- [x] Write failing tests for closing Qwen's retained VAE, model-prefixed
  checkpoint/LoRA paths overriding generic Krea settings, lazy Krea
  prefetch, and backend forwarding/closing using fake expensive loaders.
- [x] Add a public, locked, idempotent Qwen `close()` which clears `_vae`
  and unloads the runtime. Keep existing Base/Turbo behavior unchanged.
- [x] Add Qwen-prefixed model/base/LoRA directory resolution with existing
  generic settings as fallbacks. Use it consistently in catalog and loader.
- [x] Extract current handler inference adapters into model-specific
  `api.py` helpers; existing handler tests continue to exercise those helpers.
- [x] Implement Krea and Qwen backends. Krea uses `PrefetchSession` with
  explicit downloader arguments, scoped directories and pipeline parameters.
  Qwen owns the existing manager and releases it when the family changes.
- [x] Verify backend tests plus existing model suites in separate processes.

```sh
python -m unittest discover -s workers/krea2/tests -p 'test_*.py'
python -m unittest discover -s workers/qwen_image_21/tests -p 'test_*.py'
```

## Task 3: Request dispatch and clients

Files: `image_worker/api.py`, `workers/image/handler.py`, image tests,
`scripts/krea2_image.py`, `scripts/qwen_image_21.py`, existing client tests.

- [x] Write failing routing tests for explicit/default model, invalid model
  types, unknown options, model-specific limits/defaults, seed zero,
  unchanged input envelopes and response model metadata.
- [x] Select the backend profile before invoking shared `handle_job`.
  Strip only the routing field from a copied input; preserve unknown-field
  rejection. Pass the selected normalizer, error types and OOM recovery.
- [x] Keep the RunPod entrypoint thin and register the handler once.
- [x] Update both clients to submit explicit `input.model` for generate/edit;
  default Qwen to `9zb0wyo61ck3wk` with `ENDPOINT_ID` override retained.
- [x] Verify actual submitted payloads using the existing CLI test boundaries.

```json
{"input":{"model":"qwen_image_21","prompt":"a red fox","model_options":{"variant":"base"}}}
```

## Task 4: One release image and documentation

Files: `workers/image/{Dockerfile,requirements.txt,start.sh,README.md}`,
`.github/workflows/build-workers.yml`, root/shared/model documentation.

- [x] Package both runtime/download packages with isolated model directories.
  Use the agreed pins, existing Diffusers patch and component download barriers.
- [x] Verify startup without checkpoint downloads in a temporary fake app.
- [x] Change CI image matrix to `image` and `joycaption`; run shared, common
  worker and both model suites plus runtime import/dependency checks in `image`.
- [x] Make the shared API and root README describe one endpoint and model
  routing. Model README pages identify runtime backends and link common deploy.
- [ ] Complete review, commit/push, and await successful image CI.
  Dependency resolution, all 378 affected tests and whitespace checks passed locally.

## Task 5: Deploy, exercise switches, rename, retire old endpoint

- [ ] Save private before snapshots of both endpoints. Resolve and verify OCI
  source revision/digest and successful CI for the exact candidate.
- [ ] Wait for a clear retained-endpoint queue; deploy common image with scoped
  paths, original Global Volume/GPU selectors, 200 GB disk and temporary idle
  timeout. Verify full provider readback against intended changes.
- [ ] Submit Krea Identity -> Qwen Turbo -> Qwen Base -> Krea General through
  the same endpoint. Require one worker, correct model/variant/LoRA metadata,
  completed outputs, valid decoded images/hashes and matching unload/load logs.
- [ ] Inspect outputs visually; restore original idle timeout. Rename retained
  endpoint `image-workers`, then stop old Qwen with min=0/max=0 after its queue
  clears. Preserve its previous configuration as a private snapshot.
- [ ] Save sanitized deployment evidence and commands in repository docs,
  commit/push, verify final provider state and report exact test/deploy scope.
