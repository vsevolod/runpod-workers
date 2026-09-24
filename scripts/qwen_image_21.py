#!/usr/bin/env python3
"""RunPod Qwen-Image-2.1 client: text-to-image, multi-ref edit, or fetch → PNG.

Modes:
  1) Fetch:    --request-id <id>
  2) Edit:     -i/--image <path> (repeat up to 10)
  3) Generate: otherwise

Env:
  RUNPOD_API_KEY   required
  ENDPOINT_ID      required (no default endpoint yet)

Examples:
  python scripts/qwen_image_21.py --prompt "a red fox" -o out.png
  python scripts/qwen_image_21.py -i a.png -i b.png \\
    --prompt "put both subjects in one photo" -o edit.png
  python scripts/qwen_image_21.py --request-id abc123 -o out.png
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 5.0
DEFAULT_OUTPUT = "output.png"
DEFAULT_EDIT_OUTPUT = "edit_output.png"
DEFAULT_STEPS = 40
DEFAULT_GUIDANCE = 1.0
DEFAULT_GEN_SIZE = 1024
MAX_REFS = 10


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"error: set {name} in the environment", file=sys.stderr)
        raise SystemExit(1)
    return value


def _prompt_line(label: str) -> str:
    try:
        return input(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        raise SystemExit(130)


def _image_to_data_url(path: Path) -> str:
    if not path.is_file():
        print(f"error: image not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    data = path.read_bytes()
    if not data:
        print(f"error: empty image file: {path}", file=sys.stderr)
        raise SystemExit(1)
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif data[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        else:
            mime = "image/png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


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
        print(f"error: HTTP {err.code} {err.reason}: {detail[:500]}", file=sys.stderr)
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


def _redact_input_for_log(job_input: dict[str, Any]) -> dict[str, Any]:
    out = dict(job_input)
    images = out.get("images")
    if isinstance(images, list):
        out["images"] = [
            f"<{len(item)} chars>" if isinstance(item, str) and len(item) > 80 else item
            for item in images
        ]
    return out


def _submit_job(endpoint_id: str, api_key: str, job_input: dict[str, Any]) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    print(f"→ POST {url}")
    print(f"  input: {json.dumps(_redact_input_for_log(job_input), ensure_ascii=False)}")
    resp = _api_request("POST", url, api_key, {"input": job_input}, timeout=120.0)
    job_id = resp.get("id")
    if not job_id:
        print(f"error: no job id in response: {resp}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  job id: {job_id}  status: {resp.get('status') or '?'}")
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
        print(f"  [{elapsed:6.1f}s] status={status}", flush=True)
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return resp
        time.sleep(interval)


def _extract_image_bytes(output: dict[str, Any]) -> bytes:
    url = None
    images = output.get("images")
    if isinstance(images, list) and images:
        url = images[0]
    if not url:
        url = output.get("image_url")
    if not isinstance(url, str) or not url:
        raise ValueError("no images/image_url in output")
    match = re.match(r"data:image/[^;]+;base64,(.+)$", url, re.DOTALL)
    payload = match.group(1) if match else url
    return base64.b64decode(payload, validate=False)


def _save_result(result: dict[str, Any], dest: Path) -> int:
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
    try:
        raw = _extract_image_bytes(output)
    except Exception as err:
        print(f"error: could not decode image: {err}", file=sys.stderr)
        return 1
    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.write_bytes(raw)
    print(f"✓ wrote {dest} ({len(raw)} bytes)")
    for key in (
        "type",
        "seed",
        "width",
        "height",
        "num_inference_steps",
        "guidance_scale",
        "output_resolution",
        "num_refs",
    ):
        if key in output:
            print(f"  {key}: {output[key]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument(
        "--request-id",
        dest="request_id",
        default=None,
        help="Existing RunPod job id. Fetch only; do not submit.",
    )
    parser.add_argument(
        "-i",
        "--image",
        action="append",
        dest="images",
        default=None,
        help=f"Reference image. Repeat for a multi-ref edit (max {MAX_REFS}).",
    )
    parser.add_argument("--prompt", default=None, help="Prompt. Asked interactively if omitted.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--output-resolution",
        type=int,
        default=None,
        help="Condition-image area. Edit default when size is omitted: 1024.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=DEFAULT_GUIDANCE,
        help="true CFG scale. <= 1 disables guidance (default 1).",
    )
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Generate only. Edit is always 1.",
    )
    parser.add_argument("--poll-interval", type=float, default=POLL_INTERVAL_S)
    return parser


def _run_fetch(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    job_id = args.request_id.strip()
    if not job_id:
        print("error: --request-id is empty", file=sys.stderr)
        return 1
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    return _save_result(result, Path(args.output or DEFAULT_OUTPUT))


def _run_edit(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    paths = [Path(item).expanduser() for item in args.images or []]
    if not paths or len(paths) > MAX_REFS:
        print(f"error: pass 1 to {MAX_REFS} --image paths", file=sys.stderr)
        return 1
    if (args.width is None) ^ (args.height is None):
        print(
            "error: pass both --width and --height, or neither",
            file=sys.stderr,
        )
        return 1
    if args.guidance_scale > 1 and not (args.negative_prompt or "").strip():
        print(
            "error: --guidance-scale > 1 requires --negative-prompt",
            file=sys.stderr,
        )
        return 1
    prompt = args.prompt if args.prompt is not None else _prompt_line("prompt")
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 1

    job_input: dict[str, Any] = {
        "type": "image_edit",
        "prompt": prompt,
        "images": [_image_to_data_url(path) for path in paths],
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images": 1,
    }
    if args.width is not None:
        job_input["width"] = args.width
        job_input["height"] = args.height
    if args.output_resolution is not None:
        job_input["output_resolution"] = args.output_resolution
    if args.seed is not None:
        job_input["seed"] = args.seed
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt

    print(f"mode:     edit ({len(paths)} refs)")
    print(f"endpoint: {endpoint_id}")
    print(f"prompt:   {prompt}")
    job_id = _submit_job(endpoint_id, api_key, job_input)
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    return _save_result(result, Path(args.output or DEFAULT_EDIT_OUTPUT))


def _run_generate(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    if args.guidance_scale > 1 and not (args.negative_prompt or "").strip():
        print(
            "error: --guidance-scale > 1 requires --negative-prompt",
            file=sys.stderr,
        )
        return 1
    width = args.width if args.width is not None else DEFAULT_GEN_SIZE
    height = args.height if args.height is not None else DEFAULT_GEN_SIZE
    prompt = args.prompt if args.prompt is not None else _prompt_line("prompt")
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 1
    job_input: dict[str, Any] = {
        "type": "image_generate",
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images": args.num_images,
    }
    if args.output_resolution is not None:
        job_input["output_resolution"] = args.output_resolution
    if args.seed is not None:
        job_input["seed"] = args.seed
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt

    print(f"mode:     generate")
    print(f"endpoint: {endpoint_id}")
    print(f"prompt:   {prompt}")
    print(f"size:     {width}x{height}  steps={args.steps}")
    job_id = _submit_job(endpoint_id, api_key, job_input)
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    return _save_result(result, Path(args.output or DEFAULT_OUTPUT))


def main() -> int:
    args = _build_parser().parse_args()
    endpoint_id = _require_env("ENDPOINT_ID")
    api_key = _require_env("RUNPOD_API_KEY")
    if args.request_id:
        if args.images:
            print("error: --request-id cannot be combined with --image", file=sys.stderr)
            return 1
        return _run_fetch(args, endpoint_id=endpoint_id, api_key=api_key)
    if args.images:
        return _run_edit(args, endpoint_id=endpoint_id, api_key=api_key)
    return _run_generate(args, endpoint_id=endpoint_id, api_key=api_key)


if __name__ == "__main__":
    raise SystemExit(main())
