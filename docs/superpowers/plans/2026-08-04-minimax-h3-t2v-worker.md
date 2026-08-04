# MiniMax H3 T2V Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship thin RunPod Serverless worker `workers/minimax_h3/` for MiniMax H3 **T2V (t2va)** via **diffusers ModularPipeline** (pinned SHA), volume bootstrap, full bucket env for MP4 upload, unittest-only CI gate.

**Architecture:** Pure helpers in `h3_infer/` (duration, request/canvas, delivery, bucket config) are unit-tested with **unittest only**. `pipeline.py` loads MiniMaxAI weights once with 1×80GB CPU offload. `handler.py` validates input, generates MP4, uploads only when **all four** bucket env vars are set (else local_upload trap), uses `st_size` for delivery choice, `refresh_worker: true` on OOM.

**Tech Stack:** Python 3.12, torch (Dockerfile CUDA wheel), diffusers `@abc5e9bf…`, transformers, accelerate, runpod≥1.7.9, av (PyAV), huggingface-hub, unittest.

**Spec:** `docs/superpowers/specs/2026-08-04-minimax-h3-t2v-worker-design.md` (**approved v3**, canvas section aligned with packing.py in this plan rev)

**Plan rev:** 2 — blocking review fixes (bucket env, pins+av, canvas, unittest, handler OOM/size, locked APIs)

---

## File map

| File | Responsibility |
|------|----------------|
| `workers/minimax_h3/h3_infer/__init__.py` | empty or re-exports |
| `workers/minimax_h3/h3_infer/duration.py` | snap frames, duration bounds |
| `workers/minimax_h3/h3_infer/canvas.py` | `resolve_canvas_size` + `validate_canvas` (packing.py parity) |
| `workers/minimax_h3/h3_infer/request.py` | `normalize_t2v_input` → `T2VRequest` |
| `workers/minimax_h3/h3_infer/delivery.py` | bucket completeness + delivery mode |
| `workers/minimax_h3/h3_infer/pipeline.py` | load + `generate_t2v` |
| `workers/minimax_h3/schemas.py` | `INPUT_SCHEMA` |
| `workers/minimax_h3/handler.py` | RunPod entry |
| `workers/minimax_h3/download_weights.py` | HF snapshot t2va half |
| `workers/minimax_h3/requirements.txt` | **full pins below** |
| `workers/minimax_h3/Dockerfile` | CUDA image |
| `workers/minimax_h3/test_input.json` | smoke job |
| `workers/minimax_h3/NOTICE` | attributions |
| `workers/minimax_h3/LICENSES/MINIMAX-H3-COMMUNITY.txt` | full license text |
| `workers/minimax_h3/LICENSES/RUNPOD-WORKER-SDXL-MIT.txt` | copy from joycaption |
| `workers/minimax_h3/README.md` | deploy + license |
| `workers/minimax_h3/tests/test_duration.py` | unittest |
| `workers/minimax_h3/tests/test_canvas.py` | unittest |
| `workers/minimax_h3/tests/test_request.py` | unittest |
| `workers/minimax_h3/tests/test_delivery.py` | unittest |
| `workers/minimax_h3/tests/test_download_weights.py` | unittest |
| `workers/minimax_h3/tests/test_handler.py` | unittest + mocks (**required**) |
| `README.md` | monorepo row |

### Locked constants

```python
# duration.py
FPS = 24
MIN_DURATION_SEC = 5.0
MAX_DURATION_SEC = 14.375  # 345/24; last 17n+5 with output_duration <= 15

# canvas.py — from packing.py @ abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344  # 1_032_192
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4

# pipeline.py
NUM_INFERENCE_STEPS = 50
MODEL_ID = "MiniMaxAI/MiniMax-H3"
DIFFUSERS_GIT_SHA = "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"  # PR #14355 snapshot

# request defaults
DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
MAX_PROMPT_CHARS = 8000
DEFAULT_MODEL_DIR = "/runpod-volume/minimax_h3"
MAX_INLINE_VIDEO_BYTES_DEFAULT = 7_000_000

# Bucket: ALL four required for "configured" (runpod 1.7.9 get_boto_client + upload_file_to_bucket)
BUCKET_ENV_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)
```

### Official canvas sizes (`resolve_canvas_size`) — golden

| Ratio | height × width |
|-------|----------------|
| 16:9 | 768 × 1344 |
| 9:16 | 1344 × 768 |
| 1:1 | 768 × 768 |
| 4:1 | **512 × 2016** |
| 1:4 | **2016 × 512** |

Worker **must accept** `2016×512` (not reject long edge > 1344). Preview `864×480` is smaller area → allowed.

### Canvas validation rule (backend-parity intent)

Copy `resolve_canvas_size` algorithm from packing.py (same SHA). Client `width`/`height` (API uses width,height as pixel W×H; packing returns `(height, width)`):

1. Both multiples of 32; both ≥ 32.
2. `aspect = width/height` ∈ `[1/4, 4]`.
3. Let `nom_h, nom_w = resolve_canvas_size(width, height)` (ratio only).
4. Accept if `width * height <= nom_h * nom_w` (any smaller preview OK; reject larger than released canvas for that aspect).

Do **not** use `MAX_LONG_EDGE = 1344` as a side max.

### Types

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class T2VRequest:
    prompt: str
    width: int
    height: int
    requested_duration: float
    length: int
    seed: int
    output_duration: float

@dataclass(frozen=True)
class DeliveryPlan:
    mode: str  # "url" | "base64" | "error"
    error: str | None = None
```

### Test runner (locked)

**Only unittest.** Every test file uses `unittest.TestCase`. No free `def test_*` pytest functions.

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest discover -s tests -p 'test_*.py' -v
```

**Expected after Tasks 1–5 + 7 (CPU):** at least the counts listed per task; full suite **≥ 25** tests (document actual count in Task 10 when green).

### Out of scope

- I2V / R2V / `--include-ref2va` (add with R2V later)
- Comfy / SGLang backends
- Product geo implementation (README only)

---

## Locked `requirements.txt` (runtime; torch in Dockerfile)

```text
# workers/minimax_h3/requirements.txt
# Torch installed in Dockerfile from cu124 index (see Dockerfile).

runpod>=1.7.9
av>=12.0.0
numpy>=1.24.0
safetensors>=0.4.0
accelerate>=1.0.0
transformers==4.57.1
huggingface-hub>=0.26.0
# MiniMax-H3 ModularPipeline — PR #14355 commit (not yet on PyPI as of plan date)
diffusers @ git+https://github.com/huggingface/diffusers.git@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
```

- **`av` is mandatory** — `diffusers.utils.export_utils.encode_video` raises `ImportError` without PyAV.
- No `hf_transfer`. Bootstrap may set `HF_XET_HIGH_PERFORMANCE=1`.
- Dockerfile installs system libs for PyAV if needed (`ffmpeg` / libav via apt as required by `av` wheels).

---

## Locked download patterns

```python
REPO_ID = "MiniMaxAI/MiniMax-H3"

# snapshot_download allow_patterns — t2va half only (converted layout).
# Ref2VA / transformer_ref / original FL2VA trees are NOT matched → not downloaded.
ALLOW_PATTERNS_T2VA = [
    "modular_model_index.json",
    "model_index.json",
    "transformer/*",
    "transformer/**",
    "text_encoder/*",
    "text_encoder/**",
    "vae/*",
    "vae/**",
    "audio_vae/*",
    "audio_vae/**",
    "tokenizer/*",
    "tokenizer/**",
    "processor/*",
    "processor/**",
    "scheduler/*",
    "scheduler/**",
    "audio_scheduler/*",
    "audio_scheduler/**",
]
# No IGNORE_PATTERNS list in MVP (redundant if allow is exclusive).
# No --include-ref2va in MVP.
```

If a dry HF listing shows extra root json files required by `ModularPipeline.from_pretrained`, **add them to allow** in the same PR as download script — do not invent ignore noise.

---

## Locked pipeline call (t2va)

```python
# After ModularPipeline.from_pretrained + load_components(dtype=torch.bfloat16)
# + ComponentsManager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

generator = torch.Generator(device="cpu").manual_seed(req.seed)
state = pipe(
    prompt=req.prompt,
    height=req.height,
    width=req.width,
    num_frames=req.length,
    num_inference_steps=50,
    generator=generator,
)
encode_video(
    state.get("videos")[0],
    fps=24,
    output_path=output_path,
    audio=state.get("audio")[0],
    audio_sample_rate=state.get("sampling_rate"),
)
```

(kwargs match minimax-h3 docs + SetupStep: height/width/num_frames.)

---

## Locked bucket upload (runpod 1.7.9)

`get_boto_client()` needs `BUCKET_ENDPOINT_URL` + `BUCKET_ACCESS_KEY_ID` + `BUCKET_SECRET_ACCESS_KEY`. If any missing → `boto_client is None` → `upload_file_to_bucket` copies to `local_upload/` and returns a **local path**.

Without `bucket_name`, helper uses `time.strftime("%m-%y")` and does **not** create the bucket.

**Worker rules:**

```python
def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV_KEYS)

def require_bucket_or_exit() -> None:
    if os.environ.get("REQUIRE_BUCKET", "").lower() in {"1", "true", "yes"}:
        missing = [k for k in BUCKET_ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"REQUIRE_BUCKET=1 but missing: {', '.join(missing)}")

# Upload:
url = rp_upload.upload_file_to_bucket(
    file_name=f"{job_id}/output.mp4",
    file_location=str(mp4_path),
    bucket_name=os.environ["BUCKET_NAME"],
    extra_args={"ContentType": "video/mp4"},
)
# Reject if url does not look like http(s) (guards local_upload fallback)
if not str(url).startswith(("http://", "https://")):
    return {"error": "bucket upload returned non-URL path; check BUCKET_* credentials"}
```

`choose_delivery(has_bucket=bucket_configured(), raw_size=mp4_path.stat().st_size, ...)`.

---

### Task 1: duration (TDD)

**Files:** create `h3_infer/duration.py`, `tests/test_duration.py`, empty `__init__.py`s

- [ ] **Step 1: Write `tests/test_duration.py` as unittest.TestCase** (golden 5→124, 14.375→345, reject 15 and 4.99)

- [ ] **Step 2: Run discover — FAIL**

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest discover -s tests -p 'test_duration.py' -v
```

- [ ] **Step 3: Implement `snap_num_frames` / `validate_requested_duration` / `output_duration_sec`**

```python
def snap_num_frames(duration_sec: float) -> int:
    frames = max(5, round(float(duration_sec) * 24))
    return frames + (5 - (frames % 17)) % 17
```

- [ ] **Step 4: PASS** (expect **≥ 5** tests in this file)

- [ ] **Step 5: Commit** `feat(minimax_h3): duration snap and bounds`

---

### Task 2: canvas packing parity (TDD)

**Files:** create `h3_infer/canvas.py`, `tests/test_canvas.py`

- [ ] **Step 1: Tests**

```python
class TestResolveCanvas(unittest.TestCase):
    def test_16_9(self):
        h, w = resolve_canvas_size(16, 9)
        self.assertEqual((h, w), (768, 1344))

    def test_4_1(self):
        h, w = resolve_canvas_size(4, 1)
        self.assertEqual((h, w), (512, 2016))

    def test_1_4(self):
        h, w = resolve_canvas_size(1, 4)
        self.assertEqual((h, w), (2016, 512))


class TestValidateCanvas(unittest.TestCase):
    def test_accepts_preview(self):
        validate_canvas(864, 480)

    def test_accepts_4_1_official(self):
        validate_canvas(2016, 512)

    def test_rejects_not_multiple_32(self):
        with self.assertRaises(ValueError):
            validate_canvas(865, 480)

    def test_rejects_over_nominal_area(self):
        # larger than 16:9 nominal 1344x768 area
        with self.assertRaises(ValueError):
            validate_canvas(1920, 1088)
```

- [ ] **Step 2: Implement `resolve_canvas_size` byte-for-algorithm match of packing.py** + `validate_canvas(width, height)` as rules above (API width/height are W×H; compare area to `nom_h * nom_w`).

- [ ] **Step 3: PASS** (≥ 6 tests) + commit `feat(minimax_h3): canvas resolve/validate packing parity`

---

### Task 3: request normalize (TDD)

**Files:** create `h3_infer/request.py`, `tests/test_request.py`

- [ ] **Step 1: Tests** — defaults 864×480 duration 5 → length 124; seed 42; empty prompt; duration 15; non-32 size; accept 2016×512

- [ ] **Step 2: Implement `normalize_t2v_input` using duration + canvas**

- [ ] **Step 3: PASS** (≥ 6 tests) + commit

---

### Task 4: delivery + bucket_configured (TDD)

**Files:** create `h3_infer/delivery.py`, `tests/test_delivery.py`

- [ ] **Step 1: Tests as unittest.TestCase only**

```python
class TestBucketConfigured(unittest.TestCase):
    def test_all_four(self):
        env = {k: "x" for k in BUCKET_ENV_KEYS}
        self.assertTrue(bucket_configured(env))

    def test_missing_name(self):
        env = {k: "x" for k in BUCKET_ENV_KEYS}
        del env["BUCKET_NAME"]
        self.assertFalse(bucket_configured(env))

    def test_only_endpoint(self):
        self.assertFalse(bucket_configured({"BUCKET_ENDPOINT_URL": "http://s3"}))


class TestChooseDelivery(unittest.TestCase):
    def test_url_when_bucket(self):
        self.assertEqual(choose_delivery(True, 99_000_000).mode, "url")

    def test_base64_small(self):
        self.assertEqual(choose_delivery(False, 7_000_000).mode, "base64")

    def test_error_large(self):
        plan = choose_delivery(False, 7_000_001)
        self.assertEqual(plan.mode, "error")
```

- [ ] **Step 2: Implement**

```python
def bucket_configured(env: Mapping[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    return all(e.get(k) for k in BUCKET_ENV_KEYS)

def choose_delivery(has_bucket: bool, raw_size: int, max_inline: int = MAX_INLINE_VIDEO_BYTES_DEFAULT) -> DeliveryPlan:
    ...
```

- [ ] **Step 3: PASS** (≥ 6 tests) + commit

---

### Task 5: download_weights (TDD, no network)

**Files:** create `download_weights.py`, `tests/test_download_weights.py`

- [ ] **Step 1: unittest cases**

```python
class TestPatterns(unittest.TestCase):
    def test_allow_includes_transformer(self):
        self.assertTrue(any(p.startswith("transformer") for p in ALLOW_PATTERNS_T2VA))

    def test_allow_excludes_ref_by_omission(self):
        joined = " ".join(ALLOW_PATTERNS_T2VA)
        self.assertNotIn("transformer_ref", joined)
        self.assertNotIn("Ref2VA", joined)


class TestCliDryRun(unittest.TestCase):
    def test_dry_run_exit_0(self):
        rc = main(["--output", "/tmp/x", "--dry-run"])
        self.assertEqual(rc, 0)
```

Mock `snapshot_download` if testing real path branch.

CLI flags: `--output`, `--repo` (default `MiniMaxAI/MiniMax-H3`), `--token`, `--hf-home`, `--dry-run`. **No** `--include-ref2va`.

- [ ] **Step 2: Implement snapshot_download with ALLOW_PATTERNS_T2VA only**

- [ ] **Step 3: PASS** (≥ 3 tests) + commit

---

### Task 6: schemas + licenses

**Files:** `schemas.py`, `NOTICE`, `LICENSES/*`

- [ ] **Step 1: Full MiniMax H3 license text** into `LICENSES/MINIMAX-H3-COMMUNITY.txt` (curl from HF raw; no pointer)

- [ ] **Step 2: INPUT_SCHEMA**

```python
INPUT_SCHEMA = {
    "prompt": {"type": str, "required": True},
    "width": {
        "type": int,
        "required": False,
        "default": 864,
        "constraints": lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 32 and n % 32 == 0,
    },
    "height": {
        "type": int,
        "required": False,
        "default": 480,
        "constraints": lambda n: isinstance(n, int) and not isinstance(n, bool) and n >= 32 and n % 32 == 0,
    },
    "duration": {
        "type": float,
        "required": False,
        "default": 5.0,
        "constraints": lambda d: isinstance(d, (int, float)) and not isinstance(d, bool) and 5.0 <= float(d) <= 14.375,
    },
    "seed": {
        "type": int,
        "required": False,
        "default": -1,  # sentinel → random in normalize; document in README
        "constraints": lambda s: isinstance(s, int) and not isinstance(s, bool) and s >= -1,
    },
}
```

Handler maps `seed == -1` to “missing” before `normalize_t2v_input` (or accept optional by stripping -1).

- [ ] **Step 3: Commit**

---

### Task 7: handler (mocked tests required)

**Files:** `handler.py`, `tests/test_handler.py`

- [ ] **Step 1: Write required unittest with mocks** (no GPU)

Cases:

1. Missing `input` → error  
2. Valid input + mocked pipeline writes tiny mp4; no bucket → base64 key present  
3. Valid input + all four bucket env + mock `upload_file_to_bucket` returns `https://example/x.mp4` → `video_url`  
4. Bucket incomplete (only endpoint) + large file size mock → error mode (not fake url)  
5. Mock pipeline raises `torch.cuda.OutOfMemoryError` → response has `refresh_worker: True`  
6. Upload mock returns `local_upload/foo.mp4` → error (non-URL)

Use `unittest.mock.patch` on `H3Pipeline` / `rp_upload.upload_file_to_bucket`.

- [ ] **Step 2: Implement handler**

Critical behaviors:

```python
# size without full read for policy
raw_size = mp4_path.stat().st_size
plan = choose_delivery(bucket_configured(), raw_size, _max_inline())
...
if plan.mode == "base64":
    raw = mp4_path.read_bytes()  # only here
    ...
if plan.mode == "url":
    url = rp_upload.upload_file_to_bucket(
        file_name=f"{job_id}/output.mp4",
        file_location=str(mp4_path),
        bucket_name=os.environ["BUCKET_NAME"],
        extra_args={"ContentType": "video/mp4"},
    )
    if not str(url).startswith(("http://", "https://")):
        return {"error": "bucket upload returned non-URL path; check BUCKET_* credentials"}
    return {**meta, "video_url": url}

except torch.cuda.OutOfMemoryError:
    ...
    return {"error": "CUDA out of memory", "refresh_worker": True}
except Exception as err:
    ...
    return {"error": f"{type(err).__name__}: {err}", "refresh_worker": True}
```

Startup: `require_bucket_or_exit()`; construct `H3Pipeline` once.

- [ ] **Step 3: PASS handler tests** (≥ 6) + commit

---

### Task 8: pipeline module

**Files:** `h3_infer/pipeline.py`

- [ ] **Step 1: Implement `H3Pipeline`** with locked load + generate + `encode_video` (requires `av` at runtime)

- [ ] **Step 2: Fail-fast if MODEL_DIR missing `modular_model_index.json` or `transformer/`**

- [ ] **Step 3: Commit** (no GPU CI)

---

### Task 9: Dockerfile + requirements + test_input + README

**Files:** `requirements.txt` (exact content above), `Dockerfile`, `test_input.json`, `README.md`, root `README.md`

#### Dockerfile (concrete outline)

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# python3.12 + uv venv as joycaption
# apt: git, ca-certificates, and deps needed by av wheels / ffmpeg
RUN uv pip install --no-cache \
    "torch==2.7.0" "torchvision==0.22.0" \
    --extra-index-url https://download.pytorch.org/whl/cu124
COPY workers/minimax_h3/requirements.txt /app/requirements.txt
RUN uv pip install --no-cache -r /app/requirements.txt
COPY workers/minimax_h3/ /app/
# Do NOT set HF_HUB_ENABLE_HF_TRANSFER
WORKDIR /app
CMD ["python", "-u", "/app/handler.py"]
```

Build: context **repo root**, `-f workers/minimax_h3/Dockerfile`.

#### README must document

- License: Excluded Territories for **DC and users**; full Agreement path; hosted obligations  
- Bucket: all four env vars + `REQUIRE_BUCKET`  
- API fields; duration [5, 14.375]; steps fixed 50  
- Canvas: packing parity (accept 2016×512; area ≤ nominal for aspect)  
- ~144 GB weights; volume ≥200 GB  
- `download_weights.py` usage  

- [ ] **Commit**

---

### Task 10: Verification gate

- [ ] **Step 1: Unit suite**

```bash
cd workers/minimax_h3 && PYTHONPATH=. python3.12 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: **all PASS**, **zero pytest-only functions**, recorded test count **≥ 25**.

- [ ] **Step 2: Dry-run download**

```bash
python3.12 workers/minimax_h3/download_weights.py --output /tmp/minimax_h3_dry --dry-run
```

- [ ] **Step 3: Manual GPU smoke** (allowed territory): download, deploy, `test_input.json` → `length=124`, `video_url` https, audio present

- [ ] **Step 4: Do not claim production-ready without Step 3**

---

## Spec coverage

| Spec / review item | Task |
|--------------------|------|
| Backend B + SHA pin + av | 8, 9 |
| duration [5, 14.375] | 1 |
| num_inference_steps=50 | 8 |
| packing canvas (not long-edge 1344) | 2, 3 |
| Four bucket env + BUCKET_NAME | 4, 7, 9 |
| No local path as video_url | 7 |
| st_size / base64-only read | 7 |
| refresh_worker on OOM | 7 |
| unittest-only discover | 1–5, 7, 10 |
| No --include-ref2va | 5 |
| Full license text | 6 |
| License DC+users docs | 9 |

## Risks

| Risk | Mitigation |
|------|------------|
| allow_patterns miss a needed json | Dry-run + first offline from_pretrained on pod; extend allow list in same change |
| av system libs | Dockerfile apt; smoke encode |
| torch version vs diffusers SHA | Prefer torch 2.7 cu124; adjust if SHA requires otherwise and note in README |

---

## Stop / execution

Plan rev 2 addresses prior blocking review. **Do not execute Tasks until this rev is approved.**
