# MiniMax H3 ComfyUI Serverless — Implementation Plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Checkbox tracking. **Do not implement until this plan is approved.**

**Goal:** Serverless T2V via **native** ComfyUI MiniMax H3 nodes (no required custom-node pack), four Comfy-Org weights, product API → MP4 **with audio** from **SaveVideo** history only.

**Architecture:** `start.sh` runs pinned ComfyUI on `127.0.0.1:8188` + `handler.py`. Handler uses stdlib/`requests` only: inject product fields into frozen **API** workflow JSON, `POST /prompt`, poll history, resolve SaveVideo file path, deliver URL or base64. Optional SageAttention/KJNodes are **out of v1**.

**Tech Stack:** Python 3.12, `requests`, `runpod`, ComfyUI **git pin ≥ H3 (docs: 0.30.0+)**, HF hub for four files, unittest. Docker from CUDA or upgraded worker-comfyui base — **Comfy pin is authoritative**, not the worker image tag alone.

**Spec:** `docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md` (rev 2)

**Official sources:**

- Tutorial: https://docs.comfy.org/tutorials/video/minimax/minimax-h3  
- UI template: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json  
- Weights: https://huggingface.co/Comfy-Org/MiniMax-H3  
- ComfyUI H3 PR lineage: https://github.com/Comfy-Org/ComfyUI/pull/15224  
- worker-comfyui (boot pattern only): https://github.com/runpod-workers/worker-comfyui  

---

## File map (v1 — keep small)

| Path | Responsibility |
|------|----------------|
| `workers/minimax_h3_comfy/handler.py` | RunPod handler, Comfy HTTP, delivery |
| `workers/minimax_h3_comfy/workflow.py` | load API JSON, inject, snap/validate helpers |
| `workers/minimax_h3_comfy/download_weights.py` | four exact files → volume `models/` |
| `workers/minimax_h3_comfy/workflows/t2va_api.json` | frozen **API-format** export |
| `workers/minimax_h3_comfy/PINS.md` | ComfyUI commit, image base, SaveVideo history path, inject map |
| `workers/minimax_h3_comfy/start.sh` | symlink models, start Comfy, start handler |
| `workers/minimax_h3_comfy/Dockerfile` | install pinned ComfyUI + deps |
| `workers/minimax_h3_comfy/requirements.txt` | runpod, requests, huggingface_hub (bootstrap) |
| `workers/minimax_h3_comfy/test_input.json` | smoke payload |
| `workers/minimax_h3_comfy/tests/test_workflow.py` | inject: exact node ids + field values + untouched siblings |
| `workers/minimax_h3_comfy/tests/test_delivery.py` | bucket matrix |
| `workers/minimax_h3_comfy/tests/test_download_weights.py` | four filenames, no extras |
| `workers/minimax_h3_comfy/README.md` | ops |
| `workers/minimax_h3_comfy/NOTICE` + `LICENSES/` | MiniMax + Comfy |

**Not in v1:** `request.py`, `duration.py`, `canvas.py`, `delivery.py` as separate packages, `comfy_client.py`, `runtime.py`, `schemas.py` unless RunPod validator forces a tiny schema dict **inside** `handler.py`. No `httpx`, no Pydantic. No `ALLOW_RAW_WORKFLOW`.

**Four weights only:**

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

---

## Phase 1 — Headless vertical slice (blocking)

**Outcome:** On a GPU host with pinned ComfyUI + four weights on disk:

```text
API JSON → POST /prompt → /history/{prompt_id} → SaveVideo metadata → MP4 path → file has audio
```

No product handler yet. No “largest mp4” fallback.

### Steps

- [ ] **1.1 Pin ComfyUI**

Record in `PINS.md`:

| Field | Value |
|-------|--------|
| ComfyUI version/commit | must be ≥ **0.30.0** or commit containing MiniMax H3 native nodes |
| How verified | `git log` / release note / node class present |
| worker-comfyui tag (if used) | only as optional base; **must upgrade Comfy** if tag is older than H3 |

Explicit risk: worker-comfyui releases may predate H3 — treat image tag as **not sufficient**.

- [ ] **1.2 Download four weights** into Comfy `models/` layout (script may land in Phase 3; manual HF OK for Phase 1).

- [ ] **1.3 Obtain API workflow**

1. Load official T2V template (Template Library / `video_minimax_h3_t2v.json`).  
2. Ensure graph runs once in UI **or** fix until it does (still not the gate).  
3. **Export (API)** → `workers/minimax_h3_comfy/workflows/t2va_api.json`.  
4. UI subgraph template is **not** the runtime artifact.

- [ ] **1.4 Headless smoke script** (throwaway OK: `tools/headless_t2v_smoke.py` or one-off under worker)

```python
# Pseudocode — replace NODE/history keys after first successful run
import json, time, uuid, requests
from pathlib import Path

BASE = "http://127.0.0.1:8188"
wf = json.loads(Path("workflows/t2va_api.json").read_text())
# optional: inject short prompt + small canvas for speed
client_id = str(uuid.uuid4())
r = requests.post(f"{BASE}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=60)
r.raise_for_status()
prompt_id = r.json()["prompt_id"]

history = None
for _ in range(3600):
    h = requests.get(f"{BASE}/history/{prompt_id}", timeout=60).json()
    if prompt_id in h:
        history = h[prompt_id]
        break
    time.sleep(1)
assert history is not None, "timeout"

# REQUIRED: document exact path, e.g. history["outputs"]["92"]["gifs"][0] or videos/…
# DO NOT glob largest mp4
out_meta = history["outputs"]["<SAVE_VIDEO_NODE_ID>"]
# resolve to absolute path under Comfy output dir
mp4 = resolve_savevideo_path(out_meta)  # implement from real keys
assert mp4.is_file() and mp4.stat().st_size > 0
# audio: ffprobe -show_streams; expect an audio stream
```

- [ ] **1.5 Fill `PINS.md` inject + SaveVideo contract**

```markdown
## Inject map (API node ids)
| Product field | Node id | Input key |
|---------------|---------|-----------|
| prompt | … | … |
| width | … | … |
| height | … | … |
| duration or length | … | … |
| seed | … | … |

## SaveVideo history
- node id: …
- outputs key path: `outputs[<id>][<key>][0].filename` (fill real)
- type/subfolder fields: …
```

- [ ] **1.6 Record smoke metrics** (VRAM peak, wall time, canvas used) — informs default canvas.

- [ ] **1.7 Commit** pins + API JSON + smoke notes (no handler required yet)

```bash
git add workers/minimax_h3_comfy/PINS.md workers/minimax_h3_comfy/workflows/t2va_api.json
git commit -m "chore(minimax_h3_comfy): pin Comfy + freeze T2V API workflow after headless smoke"
```

**Gate:** Phase 2 **does not start** until SaveVideo history path is documented and MP4+audio verified once.

---

## Phase 2 — Minimal handler + product inject

**Files:** `handler.py`, `workflow.py`, `test_input.json`, `tests/test_workflow.py`

### Product input

```json
{
  "input": {
    "prompt": "string",
    "width": 1344,
    "height": 768,
    "duration": 5.0,
    "seed": 42
  }
}
```

- Defaults: set after Phase 1 metrics (candidates: template table 864×480 @ 0.4 MP vs native 1344×768). **Document choice.**  
- Validate: multiples of 32; enforce H3 canvas rules (short edge ≤ 768, max long edge 1344) unless Phase 1 proves template ResolutionSelector encodes more — then validate what the graph actually accepts.  
- Duration: pass seconds into the same field the API graph expects (template uses duration → math snap `17k+5` @ 24fps). Prefer **let graph snap** if API still exposes duration float.  
- Seed: `-1` → random positive int in handler.  
- **No** raw workflow field.

### `workflow.py`

```python
def load_workflow(path: Path) -> dict: ...

def inject_product(
    workflow: dict,
    *,
    prompt: str,
    width: int,
    height: int,
    duration: float,
    seed: int,
) -> dict:
    """Deep-copy; set only inject-map fields; raise if node/key missing."""
```

Inject map constants **must match** Phase 1 `PINS.md` (no placeholders).

### Handler Comfy calls (inline in `handler.py`, using `requests`)

1. `POST /prompt`  
2. Poll `/history/{prompt_id}`  
3. `mp4 = path_from_savevideo(history)` using Phase 1 contract only  
4. Return meta + path for Phase 3 delivery (or temporary local path field in tests)

### Tests — `tests/test_workflow.py` (hard, no skip)

Template **must exist** in repo after Phase 1. Tests fail if file missing (do not `skipTest`).

```python
def test_inject_sets_prompt_and_seed_exactly(self):
    wf = load_workflow(TEMPLATE)
    out = inject_product(wf, prompt="hello", width=W, height=H, duration=5.0, seed=42)
    self.assertEqual(out[PROMPT_NODE]["inputs"][PROMPT_KEY], "hello")
    self.assertEqual(out[SEED_NODE]["inputs"][SEED_KEY], 42)

def test_inject_does_not_change_unrelated_nodes(self):
    wf = load_workflow(TEMPLATE)
    out = inject_product(wf, prompt="x", width=W, height=H, duration=5.0, seed=1)
    # Pick a frozen loader node id from API JSON (UNET filename)
    self.assertEqual(
        out[UNET_NODE]["inputs"][UNET_NAME_KEY],
        wf[UNET_NODE]["inputs"][UNET_NAME_KEY],
    )

def test_inject_missing_node_raises(self):
    wf = load_workflow(TEMPLATE)
    del wf[PROMPT_NODE]
    with self.assertRaises(KeyError):
        inject_product(wf, prompt="x", width=W, height=H, duration=5.0, seed=1)
```

Replace `PROMPT_NODE` etc. with **literals from PINS**, not variables that hide mistakes.

- [ ] Implement + tests PASS  
- [ ] Commit: `feat(minimax_h3_comfy): product inject + handler prompt/history path`

---

## Phase 3 — Delivery + download_weights + CPU tests

### Delivery rules (in `handler.py`)

```python
BUCKET_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)

def bucket_state(env) -> str:
    present = [k for k in BUCKET_KEYS if env.get(k)]
    if len(present) == 0:
        return "none"
    if len(present) == 4:
        return "full"
    return "partial"  # startup must abort
```

- Worker **init**: if `partial` → log error and **exit process** (misconfiguration).  
- Job success: `full` → `upload_file_to_bucket` → `video_url`; `none` → base64 if `st_size ≤ MAX_INLINE` else clear error.  
- **Never** require bucket for all deploys.

### `download_weights.py`

```bash
python download_weights.py --output /runpod-volume/minimax_h3_comfy --dry-run
python download_weights.py --output /runpod-volume/minimax_h3_comfy
```

Writes:

```text
{output}/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
{output}/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
{output}/models/vae/minimax_h3_video_vae_fp16.safetensors
{output}/models/vae/minimax_h3_audio_vae_fp32.safetensors
```

### Tests

- `tests/test_delivery.py` — none / full / partial (partial → init error)  
- `tests/test_download_weights.py` — mock hub; assert exactly four targets  

- [ ] Commit: `feat(minimax_h3_comfy): delivery matrix + four-file bootstrap`

---

## Phase 4 — Docker / serverless smoke + measurements

### Dockerfile / start.sh

```bash
# start.sh (sketch)
set -euo pipefail
MODEL_ROOT="${MODEL_DIR:-/runpod-volume/minimax_h3_comfy}"
ln -sfn "$MODEL_ROOT/models" "$COMFYUI_PATH/models"
# start pinned ComfyUI
python "$COMFYUI_PATH/main.py" --listen 127.0.0.1 --port 8188 &
# wait until /system_stats or / OK
python -u /app/handler.py
```

- Pin ComfyUI at **Phase 1 commit** inside image build.  
- If `FROM runpod/worker-comfyui:<tag>-base`: **upgrade** Comfy tree to pin; document in README.  
- **Do not** install KJNodes / Sage for v1.

### Ops checklist

| Item | Value |
|------|--------|
| Volume | ≥ size of 4 weights + headroom (measure; expect tens of GB, not 144) |
| GPU | measure in Phase 1; start conservatively (e.g. 48 GB) until peak known |
| Env | `MODEL_DIR`, optional full `BUCKET_*` |
| Smoke | `test_input.json` → `/runsync` → MP4 with audio |

### Record in README

| Metric | Value |
|--------|--------|
| Peak VRAM | |
| Cold start | |
| Warm 5s job | |
| Default canvas | + why |

- [ ] Commit: `feat(minimax_h3_comfy): Docker entry + README smoke results`

---

## Self-review vs review comments

| Review point | Plan response |
|--------------|----------------|
| Native nodes, not mandatory custom pack | Phase 1–4; KJNodes only mentioned as non-goal |
| Task 0 = headless SaveVideo vertical slice | Phase 1 sole gate; no largest-mp4 |
| worker-comfyui base may be too old | Explicit Comfy pin ≥ 0.30.0 / H3; upgrade base if used |
| Over-modularized |  handler + workflow + download + 2–3 tests + start.sh |
| require_bucket vs base64 | partial fail start; none → inline; full → URL |
| Weak inject tests | Phase 2 exact node ids; no skipTest |
| 864×480 placeholder | default after Phase 1; official 768 short / 1344×768 native |
| No raw workflow v1 | omitted |
| Root README stale path | fixed in same change set |

---

## Out of scope reminders

- Rebuilding deleted thin `workers/minimax_h3`  
- Level-1 inject  
- R2V weights / workflows  
- SageAttention speed path  

---

## Approval note

This **rev 2** is the plan to approve or amend. Implementation starts at **Phase 1** on a GPU host with ComfyUI pin + four weights.
