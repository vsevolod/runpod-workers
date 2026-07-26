#!/usr/bin/env python3
"""RunPod JoyCaption client: image → caption (or fetch by Request ID).

Modes (auto-detected):
  1) Fetch:    --request-id <id>  → status (poll if needed) → caption text
  2) Caption:  -i/--image <path>  → caption job → text file (+ stdout)

Env:
  RUNPOD_API_KEY   API key (Bearer)  — required
  ENDPOINT_ID      optional override of hardcoded default

Examples:
  python scripts/joycaption.py -i photo.jpg
  python scripts/joycaption.py -i photo.jpg -o caption.txt --prompt "Describe briefly."
  python scripts/joycaption.py --request-id abc123 -o caption.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://api.runpod.ai/v2"
# Hardcoded serverless endpoint (override with ENDPOINT_ID env if needed).
DEFAULT_ENDPOINT_ID = "yn0krhztuguxxm"
POLL_INTERVAL_S = 2.0
DEFAULT_OUTPUT = "caption.txt"
MAX_WARN_BYTES = 7_000_000  # ~async /run practical raw-image budget before base64


def _require_api_key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        print("error: set RUNPOD_API_KEY in the environment", file=sys.stderr)
        raise SystemExit(1)
    return value


def _endpoint_id() -> str:
    return os.environ.get("ENDPOINT_ID", "").strip() or DEFAULT_ENDPOINT_ID


def _image_to_data_url(path: Path) -> str:
    if not path.is_file():
        print(f"error: image not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    data = path.read_bytes()
    if not data:
        print(f"error: empty image file: {path}", file=sys.stderr)
        raise SystemExit(1)
    if len(data) > MAX_WARN_BYTES:
        print(
            f"warning: image is {len(data)} bytes; RunPod async /run payload "
            f"limit is 10 MB (~7.5 MB raw before base64). Prefer smaller images "
            f"or /runsync-class sizes.",
            file=sys.stderr,
        )
    mime, _ = mimetypes.guess_type(str(path))
    if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            mime = "image/png"
        elif data[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        else:
            mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


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
    image = out.get("image")
    if isinstance(image, str) and len(image) > 80:
        out["image"] = f"<{len(image)} chars data URL/base64>"
    return out


def _submit_job(
    endpoint_id: str, api_key: str, job_input: dict[str, Any]
) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    print(f"→ POST {url}")
    payload = {"input": job_input}
    print(
        f"  input: {json.dumps(_redact_input_for_log(job_input), ensure_ascii=False)}"
    )
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


def _maybe_save_json(result: dict[str, Any], output: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    dump = dict(result)
    dump["output"] = output
    Path(path).write_text(
        json.dumps(dump, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  meta:   {path}")


def _save_caption(
    result: dict[str, Any],
    dest: Path,
    *,
    save_json: str | None = None,
) -> int:
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
        return 1

    caption = output.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        print(f"error: no caption in output: {output!r}", file=sys.stderr)
        return 1

    _maybe_save_json(result, output, save_json)

    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.write_text(caption.rstrip() + "\n", encoding="utf-8")

    print(f"✓ wrote {dest} ({len(caption)} chars)")
    print()
    print("--- caption ---")
    print(caption.rstrip())
    print("---------------")
    for key in ("prompt", "model"):
        if key in output:
            print(f"  {key}: {output[key]}")
    delay = result.get("delayTime")
    exec_t = result.get("executionTime")
    if delay is not None or exec_t is not None:
        print(f"  times: delay={delay}ms exec={exec_t}ms")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--image",
        default=None,
        help="Image path to caption (PNG/JPEG/WebP).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output caption text path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--request-id",
        "--job-id",
        dest="request_id",
        default=None,
        help="Existing RunPod job id: skip submit, poll/fetch caption.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Optional caption prompt override. If omitted, worker uses its "
            "default formal descriptive English prompt."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional max_new_tokens (worker default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature (worker default: 0.6)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional top_p (worker default: 0.9)",
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
        help="Optional path to write full job status JSON",
    )
    return parser


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
    return _save_caption(result, output, save_json=save_json)


def _run_caption(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    image_path = Path(args.image).expanduser()
    data_url = _image_to_data_url(image_path)

    job_input: dict[str, Any] = {"image": data_url}
    if args.prompt is not None and str(args.prompt).strip():
        job_input["prompt"] = str(args.prompt).strip()
    if args.max_new_tokens is not None:
        job_input["max_new_tokens"] = int(args.max_new_tokens)
    if args.temperature is not None:
        job_input["temperature"] = float(args.temperature)
    if args.top_p is not None:
        job_input["top_p"] = float(args.top_p)

    out = Path(args.output or DEFAULT_OUTPUT)
    print("mode:     caption")
    print(f"endpoint: {endpoint_id}")
    print(f"source:   {image_path} ({image_path.stat().st_size} bytes → data URL)")
    if "prompt" in job_input:
        print(f"prompt:   {job_input['prompt']}")
    else:
        print("prompt:   (worker default descriptive)")
    print()

    job_id = _submit_job(endpoint_id, api_key, job_input)
    print()
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    print()
    return _save_caption(result, out, save_json=args.save_json)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    endpoint_id = _endpoint_id()
    api_key = _require_api_key()
    interval = max(0.5, args.poll_interval)

    if args.request_id:
        if args.image:
            print(
                "error: --request-id cannot be combined with -i/--image",
                file=sys.stderr,
            )
            return 1
        return _run_fetch(
            endpoint_id=endpoint_id,
            api_key=api_key,
            request_id=args.request_id,
            output=Path(args.output or DEFAULT_OUTPUT),
            interval=interval,
            save_json=args.save_json,
        )

    if not args.image:
        print("error: pass -i/--image or --request-id", file=sys.stderr)
        return 1

    return _run_caption(args, endpoint_id=endpoint_id, api_key=api_key)


if __name__ == "__main__":
    raise SystemExit(main())
