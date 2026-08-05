# MiniMax H3 ComfyUI Serverless Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship RunPod Serverless worker that runs MiniMax H3 **t2va** via **headless ComfyUI**, default weights = Comfy-Org **pruned int8** DiT (~19.5 GB), product API (prompt/canvas/duration/seed), video URL or base64 delivery.

**Architecture:** Container starts pinned ComfyUI + MiniMax custom nodes; models on Network Volume under Comfy `models/`. Handler validates product input, injects into pinned API-format workflow JSON, submits to local Comfy (`127.0.0.1:8188`), waits for completion, collects video (+audio) from output, uploads via `BUCKET_*` or returns base64. Patterns from [runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui) adapted for **video** output and monorepo product schema.

**Tech Stack:** Python 3.12, RunPod serverless, ComfyUI (pinned SHA), MiniMax H3 custom nodes (pinned), huggingface_hub bootstrap, unittest, CUDA base image (or `runpod/worker-comfyui:<ver>-base` as substrate).

**Spec:** `docs/superpowers/specs/2026-08-05-minimax-h3-comfyui-serverless-design.md`

**Supersedes:** deleted `workers/minimax_h3/` thin ModularPipeline + Level-1 inject; historical plans under `2026-08-04-minimax-h3-*` are not to be implemented.

---

## File map

| Path | Responsibility |
|------|----------------|
| `workers/minimax_h3_comfy/Dockerfile` | CUDA/Comfy image; start Comfy + handler |
| `workers/minimax_h3_comfy/requirements.txt` | runpod, httpx, pydantic/stdlib only as needed |
| `workers/minimax_h3_comfy/handler.py` | RunPod entry; lifecycle Comfy client |
| `workers/minimax_h3_comfy/schemas.py` | `INPUT_SCHEMA` for product fields |
| `workers/minimax_h3_comfy/download_weights.py` | Volume bootstrap (pruned DiT + companions) |
| `workers/minimax_h3_comfy/workflows/t2va_pruned_int8_api.json` | Exported Comfy **API** workflow (node ids frozen) |
| `workers/minimax_h3_comfy/workflows/README.md` | How to re-export / update pin |
| `workers/minimax_h3_comfy/h3_comfy/__init__.py` | package |
| `workers/minimax_h3_comfy/h3_comfy/request.py` | normalize product input → `T2VRequest` |
| `workers/minimax_h3_comfy/h3_comfy/duration.py` | frames snap / duration bounds (match workflow) |
| `workers/minimax_h3_comfy/h3_comfy/canvas.py` | width/height validation (workflow packing rules) |
| `workers/minimax_h3_comfy/h3_comfy/workflow_inject.py` | deep-copy template + set prompt/seed/size/frames |
| `workers/minimax_h3_comfy/h3_comfy/comfy_client.py` | HTTP: queue, poll history, list outputs |
| `workers/minimax_h3_comfy/h3_comfy/delivery.py` | bucket completeness + url/base64/error |
| `workers/minimax_h3_comfy/h3_comfy/runtime.py` | start/wait Comfy process (if not entrypoint-managed) |
| `workers/minimax_h3_comfy/test_input.json` | smoke payload |
| `workers/minimax_h3_comfy/README.md` | volume, VRAM, env, license, smoke |
| `workers/minimax_h3_comfy/NOTICE` + `LICENSES/*` | MiniMax + Comfy attributions |
| `workers/minimax_h3_comfy/tests/test_*.py` | unit tests |

**Default artifact:**

`Comfy-Org/MiniMax-H3` → `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`

---

### Task 0: Capture pins + official workflow (research, blocking)

**Files:**
- Create: `workers/minimax_h3_comfy/PINS.md` (or section in README)
- Create: `workers/minimax_h3_comfy/workflows/t2va_pruned_int8_api.json` (after export)

- [ ] **Step 1: Document exact pins in `PINS.md`**

```markdown
# Pins (fill with real SHAs during Task 0)

| Component | Value | Source |
|-----------|-------|--------|
| ComfyUI | `<git sha or tag>` | https://github.com/comfyanonymous/ComfyUI |
| MiniMax H3 nodes package | `<repo>@<sha>` | Comfy-Org / registry used by MiniMax workflows |
| worker-comfyui reference (optional substrate) | `<tag>` | https://github.com/runpod-workers/worker-comfyui |
| Primary DiT filename | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
| TE / VAE companion files | `<list from model card>` | same repo |
| Template workflow | `workflows/t2va_pruned_int8_api.json` | Export (API) from working UI graph |
```

- [ ] **Step 2: On a GPU Pod with ComfyUI UI, load official MiniMax H3 t2va/fl2va template for pruned int8; run one local success**

Record:
- exact custom node list installed
- model file paths Comfy resolved
- peak VRAM (nvidia-smi)
- output filename pattern (mp4 / webm / png sequence)

- [ ] **Step 3: Export workflow as API JSON**

ComfyUI: **Workflow → Export (API)** → save as
`workers/minimax_h3_comfy/workflows/t2va_pruned_int8_api.json`.

- [ ] **Step 4: Write node injection map** (into `workflow_inject.py` comments + PINS)

Example shape (replace with real node ids from export):

```python
# NODE_MAP for t2va_pruned_int8 — MUST match exported JSON keys
NODE_MAP = {
    "prompt": ("6", "text"),          # CLIP/text encode positive
    "seed": ("31", "seed"),
    "width": ("27", "width"),
    "height": ("27", "height"),
    "frames_or_length": ("XX", "length"),  # whatever MiniMax node uses
}
```

- [ ] **Step 5: Commit**

```bash
git add workers/minimax_h3_comfy/PINS.md \
  workers/minimax_h3_comfy/workflows/t2va_pruned_int8_api.json \
  workers/minimax_h3_comfy/workflows/README.md
git commit -m "docs(minimax_h3_comfy): pin Comfy/nodes and freeze API workflow"
```

**Gate:** without a working local export, do not invent node ids. Stop and fix Pod setup.

---

### Task 1: Scaffold worker package + request/duration/canvas + tests

**Files:**
- Create: all `h3_comfy/*.py` stubs listed in file map (request, duration, canvas, delivery)
- Create: `tests/test_request.py`, `test_duration.py`, `test_canvas.py`, `test_delivery.py`
- Create: `schemas.py`, `test_input.json`, `NOTICE`, license copies

- [ ] **Step 1: Failing tests for duration/canvas/request (copy bounds from working workflow docs in Task 0)**

Until Task 0 freezes numbers, use **placeholders matching thin-worker defaults** and update if workflow differs:

```python
# workers/minimax_h3_comfy/tests/test_duration.py
from __future__ import annotations
import unittest
from h3_comfy.duration import snap_num_frames, validate_requested_duration, FPS

class TestDuration(unittest.TestCase):
    def test_five_seconds_frames(self):
        # Update expected after Task 0 if MiniMax pack uses different snap
        n = snap_num_frames(5.0)
        self.assertEqual(n, 124)
        self.assertEqual(FPS, 24)

    def test_rejects_below_min(self):
        with self.assertRaises(ValueError):
            validate_requested_duration(1.0)
```

```python
# workers/minimax_h3_comfy/tests/test_request.py
from h3_comfy.request import normalize_t2v_input, RequestError
import unittest

class TestRequest(unittest.TestCase):
    def test_defaults(self):
        r = normalize_t2v_input({"prompt": "a cat"})
        self.assertEqual(r.width, 864)
        self.assertEqual(r.height, 480)
        self.assertEqual(r.workflow_id, "t2va_pruned_int8")

    def test_empty_prompt(self):
        with self.assertRaises(RequestError):
            normalize_t2v_input({"prompt": "  "})
```

- [ ] **Step 2: Implement minimal `duration.py`, `canvas.py`, `request.py`, `delivery.py`**

Reuse logic patterns from deleted thin worker (duration snap `17n+5` @ 24fps if workflow matches; delivery four `BUCKET_*`).

`delivery.py` contract:

```python
BUCKET_ENV_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)
MAX_INLINE_VIDEO_BYTES_DEFAULT = 7_000_000

def bucket_configured(env=None) -> bool: ...
def choose_delivery(has_bucket: bool, raw_size: int, max_inline: int) -> DeliveryPlan: ...
```

- [ ] **Step 3: Run tests**

```bash
cd workers/minimax_h3_comfy && PYTHONPATH=. python3.12 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: PASS for Task 1 modules.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(minimax_h3_comfy): request/duration/canvas/delivery + unit tests"
```

---

### Task 2: Workflow inject (pure, no Comfy process)

**Files:**
- Create: `h3_comfy/workflow_inject.py`
- Create: `tests/test_workflow_inject.py`
- Use: `workflows/t2va_pruned_int8_api.json`

- [ ] **Step 1: Failing test — inject mutates only mapped fields**

```python
# tests/test_workflow_inject.py
from __future__ import annotations
import json
import unittest
from pathlib import Path
from h3_comfy.workflow_inject import load_template, inject_t2v

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "workflows" / "t2va_pruned_int8_api.json"

class TestInject(unittest.TestCase):
    def test_inject_prompt_seed(self):
        if not TEMPLATE.is_file():
            self.skipTest("template not frozen yet (Task 0)")
        wf = load_template(TEMPLATE)
        out = inject_t2v(
            wf,
            prompt="hello fox",
            seed=42,
            width=864,
            height=480,
            num_frames=124,
        )
        # Assert via NODE_MAP targets — example:
        # self.assertEqual(out["6"]["inputs"]["text"], "hello fox")
        self.assertIsInstance(out, dict)
        self.assertIsNot(out, wf)  # deep copy

    def test_unknown_workflow_id_raises(self):
        from h3_comfy.workflow_inject import resolve_template_path
        with self.assertRaises(ValueError):
            resolve_template_path("not_a_real_id")
```

- [ ] **Step 2: Implement inject**

```python
def load_template(path: Path) -> dict:
    return json.loads(path.read_text())

def inject_t2v(workflow: dict, *, prompt: str, seed: int, width: int, height: int, num_frames: int) -> dict:
    import copy
    wf = copy.deepcopy(workflow)
    # Apply NODE_MAP assignments; raise KeyError with clear message if node missing
    ...
    return wf
```

- [ ] **Step 3: Tests pass + commit**

```bash
git commit -m "feat(minimax_h3_comfy): inject product fields into pinned API workflow"
```

---

### Task 3: Comfy HTTP client + mocked handler path

**Files:**
- Create: `h3_comfy/comfy_client.py`
- Create: `tests/test_comfy_client.py`
- Create: `handler.py` (skeleton)
- Create: `tests/test_handler.py`

- [ ] **Step 1: Client API (mirror worker-comfyui essentials)**

```python
# h3_comfy/comfy_client.py
class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout_s: float = 3600.0): ...

    def wait_until_ready(self, deadline_s: float = 300.0) -> None:
        """Poll /system_stats or / until OK."""

    def queue_prompt(self, workflow: dict, client_id: str) -> str:
        """POST /prompt → prompt_id."""

    def wait_history(self, prompt_id: str, poll_s: float = 1.0) -> dict:
        """Poll /history/{prompt_id} until outputs present or error."""

    def collect_output_files(self, history: dict, output_dir: Path) -> list[Path]:
        """Resolve video/image filenames from history outputs; return existing paths."""
```

- [ ] **Step 2: Unit tests with `httpx`/`urllib` mocks — no live Comfy**

- [ ] **Step 3: Handler flow**

```python
# handler.py (outline)
class ModelHandler:
    def __init__(self):
        require_bucket_or_exit()
        self.client = ComfyClient()
        self.client.wait_until_ready()

def handler(job):
    req = normalize_t2v_input(job["input"])
    wf = inject_t2v(load_template(...), ...)
    prompt_id = models.client.queue_prompt(wf, client_id=job["id"])
    history = models.client.wait_history(prompt_id)
    paths = models.client.collect_output_files(history, output_dir)
    video = pick_primary_video(paths)  # prefer .mp4
    plan = choose_delivery(bucket_configured(), video.stat().st_size)
    if plan.mode == "url":
        return {**meta(req), "video_url": upload(...)}
    if plan.mode == "base64":
        return {**meta(req), "video": base64...}
    return {"error": plan.error, **meta(req)}
```

**Video collection rule:** prefer largest `.mp4` / configured SaveVideo node filename; if only frames, fail with clear error (v1 requires video node in template).

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(minimax_h3_comfy): Comfy HTTP client + handler with mocked tests"
```

---

### Task 4: Volume bootstrap script

**Files:**
- Create: `download_weights.py`
- Create: `tests/test_download_weights.py`

- [ ] **Step 1: CLI**

```bash
python download_weights.py --output /runpod-volume/minimax_h3_comfy --dry-run
python download_weights.py --output /runpod-volume/minimax_h3_comfy
```

- [ ] **Step 2: Downloads (allowlist exact files from Task 0 PINS)**

```python
COMFY_REPO = "Comfy-Org/MiniMax-H3"
PRIMARY_DIT = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
# + text encoder / vae paths from PINS — never MiniMaxAI 144GB modular pack by default
```

Layout under `--output`:

```text
{output}/models/diffusion_models/<dit>
{output}/models/text_encoders/...
{output}/models/vae/...
```

- [ ] **Step 3: Tests mock `hf_hub_download`; assert pruned filename present, no official transformer shards patterns**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(minimax_h3_comfy): bootstrap pruned Comfy weights onto volume"
```

---

### Task 5: Dockerfile + Comfy runtime entry

**Files:**
- Create: `Dockerfile`
- Create: `start.sh` (optional)
- Create: `h3_comfy/runtime.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Choose substrate**

**Option A (recommended start):** `FROM runpod/worker-comfyui:<pinned>-base` then:
- install MiniMax custom nodes at pinned SHA into `/comfyui/custom_nodes/`
- copy worker code; override CMD to start Comfy (if not already) + `python -u handler.py`

**Option B:** `FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, clone ComfyUI @ pin, pip install, same nodes.

Document chosen option in README.

- [ ] **Step 2: Env**

| Env | Meaning |
|-----|---------|
| `COMFY_HOST` | default `127.0.0.1` |
| `COMFY_PORT` | default `8188` |
| `COMFYUI_PATH` | Comfy root |
| `COMFY_MODEL_DIR` / models symlink | `/runpod-volume/minimax_h3_comfy/models` |
| `MODEL_DIR` | same volume root (alias) |
| `BUCKET_*` | four keys for URL delivery |
| `LOCAL_FILES_ONLY` | N/A for Comfy files on volume; keep if useful for HF |
| `ALLOW_RAW_WORKFLOW` | `1` enables `input.comfy_workflow` |

- [ ] **Step 3: Symlink models on start**

```bash
# start.sh sketch
ln -sfn /runpod-volume/minimax_h3_comfy/models /comfyui/models
# start ComfyUI listen 127.0.0.1:8188 in background
python /comfyui/main.py --listen 127.0.0.1 --port 8188 &
python -u /app/handler.py
```

- [ ] **Step 4: Build from monorepo root**

```bash
docker build -f workers/minimax_h3_comfy/Dockerfile -t runpod-minimax-h3-comfy .
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(minimax_h3_comfy): Dockerfile and Comfy process startup"
```

---

### Task 6: README + licenses + GPU smoke checklist

**Files:**
- Create: `README.md`, `NOTICE`, `LICENSES/MINIMAX-H3-COMMUNITY.txt`, node/Comfy license notes

- [ ] **Step 1: README sections**

1. License / Excluded Territories  
2. Network volume size (≥ 80 GB recommended) + layout  
3. Bootstrap commands  
4. Serverless env + GPU guidance (start 48 GB, measure)  
5. Product API + `test_input.json`  
6. How to update workflow pin (re-export API JSON)  
7. Explicit: **not** thin diffusers worker  

- [ ] **Step 2: Smoke checklist (operator)**

```text
1. Network Volume attach → download_weights.py
2. Deploy endpoint, MODEL_DIR=/runpod-volume/minimax_h3_comfy
3. runsync test_input.json
4. Expect video_url or video; length matches snap; seed echoed
5. Record VRAM peak and cold-start seconds in README "Status"
```

- [ ] **Step 3: Commit**

```bash
git commit -m "docs(minimax_h3_comfy): README, licenses, smoke checklist"
```

---

### Task 7: GPU validation gate (GO/NO-GO)

- [ ] **Step 1: Run one successful serverless or Pod job with pruned weights**
- [ ] **Step 2: Fill status table in README**

| Metric | Value |
|--------|-------|
| Peak VRAM | |
| Cold start | |
| Warm job time (5s 864×480) | |
| Output | mp4 yes/no audio yes/no |

- [ ] **Step 3: If fail** — document root cause (missing node, VRAM, wrong model path); do not widen scope to thin inject

- [ ] **Step 4: Commit report snippet or update design status to `validated`**

---

## Self-review vs design

| Design requirement | Task |
|--------------------|------|
| ComfyUI serverless, not ModularPipeline | Tasks 3, 5 |
| Primary pruned ~19.5 GB | Tasks 0, 4 |
| Product API prompt/size/duration/seed | Tasks 1–3 |
| Pinned workflow inject | Tasks 0, 2 |
| Video delivery URL/base64 | Tasks 1, 3 |
| Volume bootstrap | Task 4 |
| License/NOTICE | Task 6 |
| Empirical VRAM | Task 7 |
| No raw workflow as only API | Task 3 (`ALLOW_RAW_WORKFLOW` optional) |

## Execution notes

1. **Task 0 is blocking** — real API JSON + node map before inject/handler polish.  
2. Prefer monorepo worker over pure Hub deploy so video + schema stay owned.  
3. Do not reintroduce `h3_infer` diffusers pipeline or Level-1 convert code.  
4. Official MiniMaxAI 144 GB pack is out of default download.  
5. After plan approval, implement Task 0 on a GPU Pod with Comfy UI first.

---

## Ops quick reference (after implementation)

**Volume:** Network Volume ≥ 80 GB → `/runpod-volume/minimax_h3_comfy`  
**Bootstrap:** `python download_weights.py --output /runpod-volume/minimax_h3_comfy`  
**GPU:** start 48 GB class; tune down only after measured peak  
**Env:** `MODEL_DIR`, `BUCKET_*` ×4, Comfy model symlink  
**Smoke:** `test_input.json` via `/runsync`
