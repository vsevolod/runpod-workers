# JoyCaption Blackwell compatibility update, 2026-10-19

JoyCaption's older Torch build did not contain kernels for Blackwell `sm_120`.
Runpod assigned RTX PRO 6000 Blackwell MIG 24 GB workers even with the endpoint's
`AMPERE_24` selector. Weights loaded, but the SDK CUDA initialization check failed
with `no kernel image is available for execution on the device` and restarted
the worker. The supplied log contained this failure on October 11 and 19.

## Candidate image

Source commit: `db41ce435ed6f9289a8ffc6aaf10425eaad2b287`.
[Build and image validation](https://github.com/vsevolod/runpod-workers/actions/runs/39218904855)
passed, including all 12 JoyCaption helper tests, runtime imports and a compiled
`sm_120` architecture check without loading weights in CI.

```text
ghcr.io/vsevolod/runpod-workers-joycaption@sha256:3bcbc0df50c7f08d170aca1171f1d1ccb878f6be4cfae28901df711390f4db22
```

The CUDA 12.8.1 base image is pinned by digest. Shared constraints pin
Torch `2.8.0+cu128` and torchvision `0.23.0+cu128` during both Torch and application
dependency installation. Transformers remains `4.51.3`; the model, processor,
API and generation path were not changed.

## GPU validation before deployment

A dedicated Pod ran the exact published image with a Global mount containing
the existing `/workspace/joycaption` snapshot. No weights were downloaded again
or moved, and no alternative Python environment was installed. The harness
captured SDK handler registration to call the image's actual caption handler.

- Actual GPU: RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, capability 12.0.
- Runtime: Python 3.12.15, Torch 2.8.0+cu128, torchvision 0.23.0+cu128,
  CUDA build 12.8, Transformers 4.51.3.
- A CUDA bf16 matrix operation completed before model loading.
- Loading the four shards took 141.65 seconds.
- Two sequential real captions completed in 6.51 and 6.20 seconds, with
  256 maximum new tokens and seed 42. Both described the fixture correctly.
- Peak Torch allocation was 17,171,255,808 bytes (about 16 GiB).

Results, runtime metadata, the executed harness and its log were hash-verified
and persisted at `/workspace/benchmarks/2026-10-19-joycaption-cu128` on Global
volume `cmv1ht9mt000007lb00es0w9n`. The test Pod was stopped, with `EXITED` and
an absent runtime verified after saving the results.

## Production deployment and API check

Endpoint `yn0krhztuguxxm` is now version 43, using the same immutable digest.
The allowed host CUDA versions start at 12.8. The `AMPERE_24` selector, Global
volume, model environment and worker/scaling settings were preserved.
The old native Git build integration was explicitly detached so a future push
cannot replace this validated digest automatically; new releases use CI-published
images and an explicit endpoint update.

Production worker `gifl7av1ii8s9m` was also Blackwell MIG 24 GB, with the new
digest and `isStale=false`. All SDK fitness checks passed, including CUDA
initialization and matrix multiplication. Job
`360ef88f-e648-47bf-a0b2-75ee879e00e0-e1` completed successfully in 6.639 seconds
after a 273.541-second cold-start delay, returning a descriptive English caption
with the default prompt and unchanged response fields.

The [evidence](2026-10-19-joycaption-cu128/evidence.json) records runtime versions,
two standalone captions, the real production API response and verified Pod
shutdown. The [executed harness](2026-10-19-joycaption-cu128/gpu_smoke.py) is
included for reproducing the pre-deployment check; its registration interception
does not test SDK startup, which the separate production job verified.
