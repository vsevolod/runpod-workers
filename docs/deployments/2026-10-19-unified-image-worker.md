# Unified Krea and Qwen deployment — 2026-10-19

The retained endpoint `9zb0wyo61ck3wk` is named `image-workers` and serves
`input.model: "krea2"` and `input.model: "qwen_image_21"`. Omission selects Krea.
Qwen Base/Turbo remains `model_options.variant`. Both CLI clients select the
model explicitly and default to this endpoint.

## Release and validation

- Source: `0c19602ec6bfd0c1c88620c41b0faa51aa38d3b9`.
- [Successful image CI](https://github.com/vsevolod/runpod-workers/actions/runs/39246996756); both backends, shared helpers and
  common worker: 378 tests inside the release CUDA image.
- Image: `ghcr.io/vsevolod/runpod-workers-image@sha256:997a5fde6341f518b0a7e61c220edcc9597bcbb88a6ccefb94fffb35beccb469`.
- Torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8,
  Transformers 5.20.1 and Diffusers `da1d3829cf08d4f329b526d89e17cc035c049d8d`
  with the native-reference patch. Runtime imports and dependency checks passed.
- Local CPU validation: 378 tests. The exact CUDA image was then validated by CI
  and completed GPU API jobs; local CPU tests alone do not establish GPU behavior.
- Read-only code review found no Critical/Important issues.

## Live GPU sequence

All four jobs completed on worker `566ra881ejo33l` (`NVIDIA RTX A5000`)
using the exact image above, in the listed order. Each output decoded to
768×1024, had nonzero image variation, matched the requested model/variant,
LoRA, seed 42, step count and total reference budget 1 MP, and was inspected visually.
References repeat the same portrait fixture; this is a switching/adapter smoke
check, not a multi-identity or general quality benchmark.

| Case | References | Steps | API execution, seconds | Output SHA-256 |
|---|---:|---:|---:|---|
| krea-identity | 2 | 8 | 133.64 | `5a0c6b841ef3d487b8d50ff25a44c55ff9a423c0e790fe4258f662e4ff595cd3` |
| qwen-turbo | 3 | 8 | 60.96 | `a9e17fedc92b8202527557ddd3216934f91538025d70c89dfed4d905aa83c4b8` |
| qwen-base | 3 | 40 | 83.43 | `8b6dc6f06e9d2a7d758e7a13aa00d7bc711ed835e74cd7fb404f60a71f1a7ed8` |
| krea-general | 1 | 8 | 81.82 | `59b44121241dabe3c32e7cd6667c025e556f932e28d5509d3c8080f8e6c8545c` |

API execution time includes selected-checkpoint downloads/loading where needed.
Krea used Identity V2 then General V2; Qwen used `lenovo_qwen21` for Turbo and
Base. Logs show Krea → Qwen → Krea backend changes and
the same 8,650,752 allocated CUDA bytes (8.25 MiB) after unloading each family,
before its replacement loads. This residual process allocation did not grow
across the two family switches; a zero-allocation claim is not made. Qwen's
inner manager also switched Turbo → Base.

## Final provider configuration

- Common endpoint version 48, min=0/max=1,
  idle timeout 10 seconds, container disk 200 GB.
- GPU selectors, CUDA restrictions and existing Global Volume were preserved.
- Model checkpoints stay on container disk; Krea and Qwen volume LoRA directories
  remain separate and are passed through model-prefixed settings.
- Old Qwen endpoint `b1zzsd2eqqdrvp` has min=0/max=0 at version
  30. Its queue was clear before it was stopped.
- Full provider readbacks were compared against the intended configuration changes.
  Private before/after snapshots remain outside the repository.

## Diagnostic timer discovered during CI

The first candidate CI (39245778693) exposed a pre-existing Qwen
startup stack-dump SIGSEGV. Repeating its watchdog test locally reproduced ten
failing subcases across twenty test repetitions. The native asynchronous stack
walker can race with Python frame/code destruction (see
[CPython #158200](https://github.com/python/cpython/issues/158200)).

A Python watchdog now calls the direct dump API while holding the GIL and stops
with an Event plus joined thread on success/error. Native code retaining the GIL
can delay diagnostic dumps. Twenty repeated tests (forty subprocesses), the
complete Qwen suite and final image CI passed after this change.

Machine-readable job metadata, sanitized inputs, image hashes and relevant
runtime events are in [the evidence JSON](2026-10-19-unified-image-worker.json).
Image bytes, reference data URLs and credentials are excluded.
