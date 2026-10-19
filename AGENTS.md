# Image worker download and loading contract

These rules apply to every image-generation backend, including Krea, Qwen,
MOSS, and any model added later.

- Use `workers/shared/model_download.py` for Hub/cache operations and
  `workers/shared/prefetch.py` for downloader processes, component readiness,
  loading timings, failure propagation, and process cleanup. Extend these
  shared mechanisms when necessary; do not introduce a separate download
  lifecycle inside a model's inference function.
- Start a CPU-only downloader after the request selects its model/variant.
  Keep Torch/CUDA/model construction in the worker process. Imports and
  initialization may overlap downloads.
- Download components in the order the loader consumes them. Small metadata
  needed to validate the selected checkpoint may come first. Publish each
  component with `ComponentPublisher.ready` as soon as all its required files
  are complete. Include its tokenizer, configuration, scheduler, or other
  dependencies in that readiness boundary.
- Wait for an individual component immediately before loading it. While it
  loads into memory, the downloader must be free to fetch subsequent
  components. **Never wait for `all` before starting the first weight load.**
- Include required built-in adapters in this same lifecycle. For example,
  a cold MOSS Turbo request prefetches its DMD LoRA while the SFT components
  load; Base does not fetch that adapter. A warm switch that needs only an
  adapter still uses the shared downloader and local cache.
- Load prepared weights with `local_files_only=True`. No Hub download may
  happen inside denoising, prompt encoding, or adapter activation. Additional
  user LoRAs use `shared.adapter_cache.AdapterCache` to stage only selected
  files from the attached volume to container disk before reading weights.
- Preserve revision pinning, expected file-size validation, atomic manifests,
  and warm-cache reuse. Publish readiness only after validation. A retry must
  reuse completed components and repair missing/truncated files. Components
  from one repository must retain the same pinned revision across a retry;
  never combine cached weights with other components fetched from a newer `main`.
- Before serving a result, wait for final `all` readiness and successful
  downloader exit. A later adapter/download failure must fail the request,
  even if earlier components loaded successfully. Terminate/reap the child
  process on loading failure and release partially loaded model resources.
- Preserve each model's precision, mmap behavior, CPU offload, scheduler,
  adapter scope/cleanup, and numerical settings when changing I/O scheduling.

Verify changes with controlled downloader subprocesses that prove an early
component loads before later downloads finish. Cover later-download failures,
partial files, retry/cache reuse, and the selected adapter/variant. Do not
replace those checks with assertions about source-code text or sleep-based
performance thresholds. Run all affected image-worker suites separately:

```sh
python -m unittest discover -s workers/shared/tests -p 'test_*.py'
python -m unittest discover -s workers/image/tests -p 'test_*.py'
python -m unittest discover -s workers/krea2/tests -p 'test_*.py'
python -m unittest discover -s workers/qwen_image_21/tests -p 'test_*.py'
python -m unittest discover -s workers/moss_image/tests -p 'test_*.py'
```

Report source/test evidence separately from a built image, deployment,
real-GPU generation, and measured cold-start improvement. Do not launch a
GPU calibration merely to verify changes to the download orchestration.
