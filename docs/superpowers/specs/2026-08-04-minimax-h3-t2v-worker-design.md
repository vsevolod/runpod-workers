# SUPERSEDED — MiniMax H3 thin T2V worker (diffusers)

**Status:** **SUPERSEDED 2026-08-05**

This design targeted a thin RunPod worker on **diffusers ModularPipeline** +
official MiniMaxAI ~144 GB pack (and later Level-1 Comfy DiT inject without
ComfyUI).

**Replacement:** ComfyUI serverless worker design:

- `docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`
- `docs/superpowers/plans/2026-08-05-minimax-h3-comfyui-serverless.md`

**Code:** `workers/minimax_h3/` **deleted** 2026-08-05. Do not re-implement this
spec unless product explicitly reverts from ComfyUI.

Original body removed from active docs tree to avoid agent confusion; recover from
git history before the deletion commit if needed.
