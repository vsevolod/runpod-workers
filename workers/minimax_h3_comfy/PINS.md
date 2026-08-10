# MiniMax H3 ComfyUI worker — pins & contracts

## ComfyUI pin

| Field | Value |
|-------|--------|
| Tag | `v0.30.0` |
| Commit | `b1693ecba9f5b65f8c80ab36b195ab963ec92413` |
| Why | Official docs: MiniMax H3 native nodes require ComfyUI **≥ 0.30.0** (PR lineage #15224) |
| How verified | Release `v0.30.0` on GitHub; `SaveVideo` / `CreateVideo` in `comfy_extras/nodes_video.py`; H3 nodes ship with this tag |
| worker-comfyui tag | **Not sufficient alone** — if used as base, **overwrite** ComfyUI tree to this pin |

Repo: https://github.com/Comfy-Org/ComfyUI

## Four weights (T2V only)

Relative paths (under HF snapshot **or** under volume `models/`):

```text
diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
vae/minimax_h3_video_vae_fp16.safetensors
vae/minimax_h3_audio_vae_fp32.safetensors
```

| Source | ID / path |
|--------|-----------|
| Full HF repo (all variants, ~465 GB) | `Comfy-Org/MiniMax-H3` — contains the four T2V files plus many others; **not** default Model Cache target |
| Slim HF repo (G0 path A, ~42.5 GB) | **TBD** after publish (e.g. `YOUR_ORG/MiniMax-H3-T2V-comfy-four`) — **preferred** for RunPod Model field |
| Legacy volume layout | `{MODEL_DIR}/models/<paths above>` |
| Materialize root (worker) | hard-coded **`/models`** (symlinks only) |
| RunPod HF cache | `/runpod-volume/huggingface-cache/hub/models--{org}--{name}/snapshots/<hash>/` |

Env: `MODEL_NAME` = endpoint **Model** field (`org/name`, main only). Interim default `Comfy-Org/MiniMax-H3` for path B experiments.

Shared constant: `download_weights.WEIGHTS` / `model_store.WEIGHT_RELS`.
`download_weights.py` = **operator CLI** for offline volume fill only (not serverless boot).

## Workflow artifact

| Item | Value |
|------|--------|
| Source template | [video_minimax_h3_t2v.json](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json) (UI + subgraph) |
| Runtime file | `workflows/t2va_api.json` — **API format**, subgraph flattened to native nodes |
| Export note | Flattened from official template node ids; not UI subgraph export as-is |

## Inject map (API node ids)

| Product field | Node id | `class_type` | Input key |
|---------------|---------|--------------|-----------|
| prompt | `104` | MiniMaxH3ImageToVideo | `prompt` |
| width | `104` | MiniMaxH3ImageToVideo | `width` |
| height | `104` | MiniMaxH3ImageToVideo | `height` |
| duration (seconds) | `111` | PrimitiveFloat | `value` |
| seed | `15` | RandomNoise | `noise_seed` |

Duration → frame length snap stays in graph node `107` (`ComfyMathExpression`, formula `17k+5` @ 24fps). Do **not** inject `length` on `104` from the handler unless the math node is removed.

Frozen model filenames (must not change on inject):

| Node | Key | Filename |
|------|-----|----------|
| `6` | `unet_name` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| `13` | `clip_name` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `11` | `vae_name` | `minimax_h3_video_vae_fp16.safetensors` |
| `24` | `vae_name` | `minimax_h3_audio_vae_fp32.safetensors` |

## SaveVideo history contract

From ComfyUI `v0.30.0` source (`comfy_extras/nodes_video.py` + `comfy_api/latest/_ui.py`):

- `SaveVideo.execute` returns `ui.PreviewVideo([SavedResult(file, subfolder, FolderType.output)])`
- `PreviewVideo.as_dict()` → `{"images": [<SavedResult>…], "animated": (True,)}`

So history path is:

```text
history[prompt_id]["outputs"]["92"]["images"][0]["filename"]
history[prompt_id]["outputs"]["92"]["images"][0]["subfolder"]
history[prompt_id]["outputs"]["92"]["images"][0]["type"]   # "output"
```

| Field | Value |
|-------|--------|
| SaveVideo node id | `92` |
| outputs key | **`images`** (not `gifs` / `videos`) |
| entry shape | `{filename, subfolder, type}` |
| on-disk root | Comfy `output/` directory when `type == "output"` |
| filename_prefix (template) | `video/MiniMax_H3` → subfolder typically `video/MiniMax_H3` |

**Forbidden:** glob largest `.mp4` under `output/`.

### GPU smoke status

| Item | Status |
|------|--------|
| Source-derived history keys | documented above (authoritative for v0.30.0) |
| Live POST `/prompt` → MP4+audio | **pending** — needs GPU ≥~24–48 GB (local host is RTX 3070 Ti 8 GB; insufficient for pruned int8 pack) |
| Smoke script | `tools/headless_t2v_smoke.py` |
| Metrics (VRAM / wall / canvas) | fill after first successful smoke |

## Product defaults (provisional until smoke metrics)

| Field | Default | Rationale |
|-------|---------|-----------|
| width × height | **864 × 480** | Official template ResolutionSelector 0.4 MP 16:9 (preview cost). Revisit after smoke vs 1344×768 native canvas. |
| duration | **5.0** s | Template outer default |
| seed | `-1` → random in handler | |

Canvas rules: multiples of 32; short edge ≤ 768; long edge ≤ 1344 when enforcing native H3 canvas.
