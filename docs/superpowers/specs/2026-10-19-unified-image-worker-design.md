# Unified image worker

## Accepted scope

Combine Krea 2 and Qwen Image 2.1 in one image worker and one RunPod endpoint.
The user selected existing Krea endpoint `9zb0wyo61ck3wk` to preserve its ID,
and requested renaming it in RunPod. Its final name is `image-workers`.
After successful verification, stop the old Qwen endpoint `b1zzsd2eqqdrvp`.
Switching latency is explicitly outside the decision criteria for this change.

## Request and response

`input.model` accepts `krea2` or `qwen_image_21`; omission defaults to Krea
so existing requests to the retained endpoint continue to work.
The selected model is returned as `output.model`. Qwen Base/Turbo remains
`model_options.variant`, with existing defaults, validation and metadata.
Existing common fields and model-specific options retain their meanings.
Unknown models and fields are rejected before loading or replacing a model.
Both bundled clients send an explicit model and default to the retained ID.

## Runtime ownership

One registry maps model IDs to backends. Each backend owns loading,
inference and closing its model. One manager serializes complete inference
calls and releases the previous backend before loading another.
Initialization imports code but neither downloads nor loads a checkpoint.
Krea downloads on first selection and reuses its container-disk cache.
Qwen retains its existing Base/Turbo manager, file-completeness checks,
local repair and shared texture-fix VAE within the Qwen backend.
Closing Qwen also releases that shared VAE before Krea can load.
Failed backend initialization clears manager ownership and permits retry.
CUDA caches are released after the old backend's references are removed.

LoRAs remain model-specific. Krea uses `/runpod-volume/krea2/loras`; Qwen
uses `/runpod-volume/qwen_image_21_turbo/loras`. Both endpoints already
mount Global Volume `cmv1ht9mt000007lb00es0w9n` at `/runpod-volume`.
Selected adapters are still staged through the existing local adapter cache.
Krea built-in V2 adapters remain version-pinned local downloads.

Model-scoped environment settings prevent Krea's existing `MODEL_DIR` and
`LORA_DIR` from selecting Qwen's checkpoint or adapters. Qwen configuration
adds prefixed paths while preserving generic fallbacks for existing helpers.
No per-request mutation of process environment is used.

## Image and build

Publish `runpod-workers-image` from `workers/image/Dockerfile`.
Use CUDA 12.8.1, Torch `2.8.0+cu128`, torchvision `0.23.0+cu128`,
Transformers `5.20.1` and Diffusers commit
`da1d3829cf08d4f329b526d89e17cc035c049d8d`.
Apply the existing Qwen native-reference patch to that pinned Diffusers.
The image includes both inference packages, downloaders, notices and licenses.
CI builds the common image and JoyCaption; model folders remain runtime
backends rather than separate image releases.

## Release and proof

Run common-manager, routing, backend and existing model/shared tests,
then the same checks in the published CUDA image. Deploy by immutable
digest only after CI succeeds and the existing endpoint queue is clear.
Keep min=0/max=1 and the current GPU/volume settings. Increase container
disk to 200 GB for the local Krea/Base/Turbo caches. Temporarily retain
the worker long enough to run Krea -> Qwen Turbo -> Qwen Base -> Krea.
Verify all outputs, hashes, response model/variant, actual LoRA attachment,
worker/image identity and unload/load events on one GPU worker.
Restore the original idle timeout, rename the retained endpoint, then
set the old Qwen endpoint to min=0/max=0 after its queue is also clear.
Record configuration differences and evidence without credentials or images.
