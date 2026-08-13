#!/usr/bin/env python3
"""RunPod MiniMax H3 T2V client: submit smoke job or fetch by Request ID → MP4.

Modes (auto-detected):
  1) Health:   --health            → GET endpoint health (workers / jobs)
  2) Fetch:    --request-id <id>   → status (poll if needed) → MP4
  3) Generate: otherwise           → T2V job → MP4

Use after deploying model_store (Model Cache or legacy volume). A successful
job with video proves V3; inspect container logs for [ModelStore] source=cache|volume
to confirm V2 / V4.1 boot path.

I2V / FL2V (same endpoint, same fl2va weights):
  --first-image path|URL|data-URL  → product first_image
  --last-image  path|URL|data-URL  → product last_image (requires first)

Env:
  RUNPOD_API_KEY   API key (Bearer) — required
  ENDPOINT_ID      serverless endpoint id — required unless --endpoint

Examples:
  # Short smoke (default canvas 864×480, duration 2s, fixed seed)
  export RUNPOD_API_KEY=...
  export ENDPOINT_ID=your_endpoint_id
  python scripts/minimax_h3_t2v.py -o smoke.mp4

  python scripts/minimax_h3_t2v.py \\
    --prompt "A calm empty room, soft daylight" \\
    --width 864 --height 480 --duration 2 --seed 42 \\
    -o smoke.mp4 --save-json smoke.json

  # Already submitted job
  python scripts/minimax_h3_t2v.py --request-id abc123 -o out.mp4

  # Workers ready?
  python scripts/minimax_h3_t2v.py --health
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 5.0  # video jobs are slow; poll less aggressively than image workers
DEFAULT_OUTPUT = "minimax_h3_smoke.mp4"
DEFAULT_PROMPT = (
    "A calm empty room, soft daylight, slow camera drift, subtle ambient audio. "
    "No text, no logos."
)
DEFAULT_WIDTH = 864
DEFAULT_HEIGHT = 480
DEFAULT_DURATION = 2.0  # short smoke (template product default is 5.0)
DEFAULT_SEED = 42


def _require_api_key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        print("error: set RUNPOD_API_KEY in the environment", file=sys.stderr)
        raise SystemExit(1)
    return value


def _endpoint_id(cli: str | None = None) -> str:
    value = (cli or "").strip() or os.environ.get("ENDPOINT_ID", "").strip()
    if not value:
        print(
            "error: set ENDPOINT_ID env or pass --endpoint <id>",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def _api_request(
    method: str,
    url: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        print(f"error: HTTP {err.code} {err.reason}: {detail[:800]}", file=sys.stderr)
        raise SystemExit(1) from err
    except urllib.error.URLError as err:
        print(f"error: request failed: {err.reason}", file=sys.stderr)
        raise SystemExit(1) from err
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        print(f"error: invalid JSON response: {err}", file=sys.stderr)
        raise SystemExit(1) from err


def _submit_job(
    endpoint_id: str, api_key: str, job_input: dict[str, Any]
) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    print(f"→ POST {url}")
    payload = {"input": job_input}
    log_input = dict(job_input)
    for key in ("first_image", "last_image"):
        val = log_input.get(key)
        if isinstance(val, str) and len(val) > 120:
            log_input[key] = f"<{len(val)} chars>"
    print(f"  input: {json.dumps(log_input, ensure_ascii=False)}")
    resp = _api_request("POST", url, api_key, payload, timeout=120.0)
    job_id = resp.get("id")
    status = resp.get("status")
    if not job_id:
        print(f"error: no job id in response: {resp}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  job id: {job_id}  status: {status or '?'}")
    return str(job_id)


def _poll_job(
    endpoint_id: str, api_key: str, job_id: str, *, interval: float
) -> dict[str, Any]:
    url = f"{API_BASE}/{endpoint_id}/status/{job_id}"
    print(f"→ polling {url} every {interval:g}s")
    started = time.monotonic()
    while True:
        resp = _api_request("GET", url, api_key, timeout=30.0)
        status = resp.get("status") or "?"
        elapsed = time.monotonic() - started
        delay = resp.get("delayTime")
        exec_t = resp.get("executionTime")
        extras: list[str] = []
        if delay is not None:
            extras.append(f"delay={delay}ms")
        if exec_t is not None:
            extras.append(f"exec={exec_t}ms")
        extra_s = f" ({', '.join(extras)})" if extras else ""
        print(f"  [{elapsed:6.1f}s] status={status}{extra_s}", flush=True)

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return resp

        time.sleep(interval)


def _download_url(url: str, *, timeout: float = 300.0) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "*/*"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        print(
            f"error: download HTTP {err.code} {err.reason}: {detail[:300]}",
            file=sys.stderr,
        )
        raise SystemExit(1) from err
    except urllib.error.URLError as err:
        print(f"error: download failed: {err.reason}", file=sys.stderr)
        raise SystemExit(1) from err


def _extract_video_bytes(output: dict[str, Any]) -> tuple[bytes, str]:
    """Return (mp4_bytes, source_label) from worker output."""
    video_url = output.get("video_url")
    if isinstance(video_url, str) and video_url.strip():
        url = video_url.strip()
        if url.startswith("data:video"):
            return _decode_data_url_or_b64(url), "video_url(data)"
        print(f"  downloading video_url ({len(url)} chars)…")
        return _download_url(url), "video_url"

    video = output.get("video")
    if isinstance(video, str) and video.strip():
        return _decode_data_url_or_b64(video.strip()), "video(base64)"

    raise ValueError(
        "no video_url, video, or s3 bucket/key in output "
        f"(keys={list(output.keys())}; delivery={output.get('delivery')!r})"
    )


def _decode_data_url_or_b64(raw: str) -> bytes:
    match = re.match(r"data:video/[^;]+;base64,(.+)$", raw, re.DOTALL)
    b64 = match.group(1) if match else raw
    # Strip whitespace/newlines some transports inject
    b64 = re.sub(r"\s+", "", b64)
    return base64.b64decode(b64, validate=False)


def _maybe_save_json(
    result: dict[str, Any], output: dict[str, Any], path: str | None
) -> None:
    if not path:
        return
    dump = dict(result)
    out_copy = dict(output)
    for key in ("video", "video_url"):
        val = out_copy.get(key)
        if isinstance(val, str) and len(val) > 120:
            out_copy[key] = f"<{len(val)} chars>"
    dump["output"] = out_copy
    Path(path).write_text(
        json.dumps(dump, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  meta:   {path}")


def _looks_like_mp4(data: bytes) -> bool:
    # ftyp box usually within first bytes; allow small offset for some writers
    if len(data) < 12:
        return False
    if data[4:8] == b"ftyp":
        return True
    return b"ftyp" in data[:64]


def _print_output_meta(result: dict[str, Any], output: dict[str, Any]) -> None:
    for key in (
        "delivery",
        "bucket",
        "key",
        "bytes",
        "width",
        "height",
        "duration",
        "seed",
        "model",
        "mode",
        "filename",
        "prompt_id",
    ):
        if key in output:
            print(f"  {key}: {output[key]}")
    delay = result.get("delayTime")
    exec_t = result.get("executionTime")
    if delay is not None or exec_t is not None:
        print(f"  times: delay={delay}ms exec={exec_t}ms")


def _save_result(
    result: dict[str, Any],
    dest: Path,
    *,
    save_json: str | None = None,
) -> int:
    """Decode job result and write MP4. Returns process exit code."""
    status = result.get("status")
    if status != "COMPLETED":
        err = result.get("error") or result.get("output") or result
        print(f"error: job {status}: {err}", file=sys.stderr)
        return 1

    output = result.get("output")
    if not isinstance(output, dict):
        print(f"error: unexpected output: {output!r}", file=sys.stderr)
        return 1

    if output.get("error"):
        print(f"error: worker: {output['error']}", file=sys.stderr)
        if output.get("traceback"):
            print(output["traceback"], file=sys.stderr)
        return 1

    _maybe_save_json(result, output, save_json)

    if output.get("delivery") == "s3" or (
        output.get("bucket") and output.get("key")
    ):
        print("✓ S3 object (consumer downloads with its own credentials)")
        _print_output_meta(result, output)
        print()
        print("Not writing a local MP4 — no presigned URL. GetObject:")
        print(f"  bucket={output.get('bucket')!s}")
        print(f"  key={output.get('key')!s}")
        return 0

    try:
        raw, source = _extract_video_bytes(output)
    except Exception as err:
        print(f"error: could not decode video: {err}", file=sys.stderr)
        return 1

    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)

    print(f"✓ wrote {dest} ({len(raw)} bytes, via {source})")
    if not _looks_like_mp4(raw):
        print(
            "warning: payload does not look like MP4 (missing ftyp); "
            "open manually / run ffprobe",
            file=sys.stderr,
        )
    else:
        print("  container: looks like MP4 (ftyp present)")

    _print_output_meta(result, output)

    print()
    print("Next (optional):")
    print(f"  ffprobe -hide_banner {dest}")
    print(
        "  # Container logs should show [ModelStore] source=cache|volume "
        "and ok weight lines with GB sizes"
    )
    return 0


def _run_health(*, endpoint_id: str, api_key: str) -> int:
    url = f"{API_BASE}/{endpoint_id}/health"
    print("mode:     health")
    print(f"endpoint: {endpoint_id}")
    print(f"→ GET {url}")
    resp = _api_request("GET", url, api_key, timeout=30.0)
    print(json.dumps(resp, indent=2, ensure_ascii=False))

    # RunPod shape varies; surface common fields if present.
    workers = resp.get("workers") if isinstance(resp, dict) else None
    jobs = resp.get("jobs") if isinstance(resp, dict) else None
    if isinstance(workers, dict):
        ready = workers.get("ready") or workers.get("idle") or workers.get("running")
        print(f"  workers summary: {workers}")
        if ready == 0 or (
            isinstance(workers.get("ready"), int) and workers["ready"] == 0
            and isinstance(workers.get("running"), int)
            and workers["running"] == 0
        ):
            print(
                "warning: no ready/running workers yet "
                "(cold start / unhealthy / scale-to-zero)",
                file=sys.stderr,
            )
    if isinstance(jobs, dict):
        print(f"  jobs summary: {jobs}")
    return 0


def _run_fetch(
    *,
    endpoint_id: str,
    api_key: str,
    request_id: str,
    output: Path,
    interval: float,
    save_json: str | None,
) -> int:
    job_id = request_id.strip()
    if not job_id:
        print("error: --request-id is empty", file=sys.stderr)
        return 1
    print("mode:     fetch")
    print(f"endpoint: {endpoint_id}")
    print(f"request:  {job_id}")
    print()
    result = _poll_job(endpoint_id, api_key, job_id, interval=interval)
    print()
    return _save_result(result, output, save_json=save_json)


def _image_ref_for_job(value: str) -> str:
    """Path → data URL; http(s) and data: pass through."""
    text = value.strip()
    if not text:
        raise ValueError("empty image ref")
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("data:"):
        return text
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(f"image file not found: {path}")
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _run_generate(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    prompt = args.prompt if args.prompt is not None else DEFAULT_PROMPT
    if not str(prompt).strip():
        print("error: prompt is required", file=sys.stderr)
        return 1

    width = args.width if args.width is not None else DEFAULT_WIDTH
    height = args.height if args.height is not None else DEFAULT_HEIGHT
    duration = args.duration if args.duration is not None else DEFAULT_DURATION
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    if width % 32 != 0 or height % 32 != 0:
        print(
            f"error: width/height must be multiples of 32 (got {width}x{height})",
            file=sys.stderr,
        )
        return 1
    if duration <= 0:
        print("error: duration must be positive", file=sys.stderr)
        return 1

    first_image = None
    last_image = None
    try:
        if args.first_image:
            first_image = _image_ref_for_job(args.first_image)
        if args.last_image:
            last_image = _image_ref_for_job(args.last_image)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if last_image and not first_image:
        print("error: --last-image requires --first-image", file=sys.stderr)
        return 1

    job_input: dict[str, Any] = {
        "prompt": str(prompt).strip(),
        "width": int(width),
        "height": int(height),
        "duration": float(duration),
        "seed": int(seed),
    }
    if first_image:
        job_input["first_image"] = first_image
    if last_image:
        job_input["last_image"] = last_image

    mode_label = "T2V"
    if first_image and last_image:
        mode_label = "FL2V"
    elif first_image:
        mode_label = "I2V"

    out = Path(args.output or DEFAULT_OUTPUT)
    print(f"mode:     generate ({mode_label} smoke)")
    print(f"endpoint: {endpoint_id}")
    print(f"prompt:   {job_input['prompt'][:120]}{'…' if len(job_input['prompt']) > 120 else ''}")
    print(
        f"canvas:   {width}x{height}  duration={duration}s  seed={seed}"
    )
    if first_image:
        src = args.first_image
        print(f"first:    {src[:80]}{'…' if len(str(src)) > 80 else ''}")
    if last_image:
        src = args.last_image
        print(f"last:     {src[:80]}{'…' if len(str(src)) > 80 else ''}")
    print(f"output:   {out}")
    print()
    print(
        "note: cold start may take minutes (Model Cache / Comfy / weights). "
        "Watch RunPod Container logs for [ModelStore] and ComfyUI ready."
    )
    print()

    # Avoid dumping multi-MB base64 to console
    log_input = dict(job_input)
    for key in ("first_image", "last_image"):
        val = log_input.get(key)
        if isinstance(val, str) and len(val) > 120:
            log_input[key] = f"<{len(val)} chars>"
    print(f"→ POST job input keys: {list(job_input.keys())}")
    print(f"  input summary: {json.dumps(log_input, ensure_ascii=False)[:500]}")

    job_id = _submit_job(endpoint_id, api_key, job_input)
    print()
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(1.0, args.poll_interval)
    )
    print()
    return _save_result(result, out, save_json=args.save_json)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="RunPod serverless endpoint id (default: ENDPOINT_ID env)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output MP4 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--request-id",
        "--job-id",
        dest="request_id",
        default=None,
        help="Existing RunPod job id: skip submit, poll/fetch MP4",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Only GET /health for the endpoint (no job)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=f"T2V prompt (default: short smoke prompt from test_input.json)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=f"Width, multiple of 32 (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Height, multiple of 32 (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--first-image",
        default=None,
        help="I2V first frame: local path, https URL, or data: URL",
    )
    parser.add_argument(
        "--last-image",
        default=None,
        help="Optional last frame (requires --first-image)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help=f"Duration seconds (default: {DEFAULT_DURATION} for smoke)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Seed; -1 = worker random (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help=f"Status poll interval in seconds (default: {POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to write job status JSON (video fields redacted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    api_key = _require_api_key()
    endpoint_id = _endpoint_id(args.endpoint)

    if args.health:
        return _run_health(endpoint_id=endpoint_id, api_key=api_key)

    if args.request_id:
        return _run_fetch(
            endpoint_id=endpoint_id,
            api_key=api_key,
            request_id=args.request_id,
            output=Path(args.output or DEFAULT_OUTPUT),
            interval=max(1.0, args.poll_interval),
            save_json=args.save_json,
        )

    return _run_generate(args, endpoint_id=endpoint_id, api_key=api_key)


if __name__ == "__main__":
    raise SystemExit(main())
