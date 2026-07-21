#!/usr/bin/env python3
"""Interactive RunPod Krea2 client: prompt + LoRA → poll → output.png."""

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
POLL_INTERVAL_S = 2.0
DEFAULT_OUTPUT = "output.png"


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"error: set {name} in the environment", file=sys.stderr)
        raise SystemExit(1)
    return value


def _prompt_line(label: str, *, default: str | None = None) -> str:
    if default is not None:
        suffix = f" [{default}]"
    else:
        suffix = ""
    try:
        raw = input(f"{label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        raise SystemExit(130)
    if not raw and default is not None:
        return default
    return raw


def _parse_lora(raw: str) -> list[dict[str, Any]]:
    """Parse 'name', 'name@1.2', or empty → []. Multiple: comma-separated."""
    raw = raw.strip()
    if not raw:
        return []
    items: list[dict[str, Any]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "@" in part:
            name, strength_s = part.rsplit("@", 1)
            name = name.strip()
            try:
                strength = float(strength_s.strip())
            except ValueError:
                print(f"error: bad LoRA strength in {part!r}", file=sys.stderr)
                raise SystemExit(1)
        else:
            name, strength = part, 1.0
        if not name:
            print(f"error: empty LoRA name in {part!r}", file=sys.stderr)
            raise SystemExit(1)
        # Catalog ID = filename stem; strip accidental suffix/path.
        name = Path(name).name
        if name.endswith(".safetensors"):
            name = name[: -len(".safetensors")]
        items.append({"name": name, "strength": strength})
    if len(items) > 4:
        print("error: at most 4 LoRAs allowed", file=sys.stderr)
        raise SystemExit(1)
    return items


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


def _submit_job(
    endpoint_id: str, api_key: str, job_input: dict[str, Any]
) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    print(f"→ POST {url}")
    payload = {"input": job_input}
    print(f"  input: {json.dumps(job_input, ensure_ascii=False)}")
    resp = _api_request("POST", url, api_key, payload)
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
    last_status = None
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
        # Heartbeat every poll so the terminal never looks frozen.
        print(f"  [{elapsed:6.1f}s] status={status}{extra_s}", flush=True)

        if status != last_status:
            last_status = status

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
    b64 = match.group(1) if match else url
    return base64.b64decode(b64, validate=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
        help=f"Status poll interval in seconds (default: {POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="Image width (default: 1024)"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="Image height (default: 1024)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional seed"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="num_inference_steps (default: 8)",
    )
    args = parser.parse_args()

    endpoint_id = _require_env("ENDPOINT_ID")
    api_key = _require_env("RUNPOD_API_KEY")

    print(f"endpoint: {endpoint_id}")
    print()

    prompt = _prompt_line("prompt")
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 1

    lora_raw = _prompt_line(
        "lora (name or name@strength, comma-separated; empty = none)",
        default="",
    )
    loras = _parse_lora(lora_raw)

    job_input: dict[str, Any] = {
        "prompt": prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": 0.0,
        "mu": 1.15,
        "loras": loras,
    }
    if args.seed is not None:
        job_input["seed"] = args.seed

    print()
    job_id = _submit_job(endpoint_id, api_key, job_input)
    print()
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    print()

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

    dest = Path(args.output)
    if not dest.is_absolute():
        dest = Path.cwd() / dest
    dest.write_bytes(raw)

    print(f"✓ wrote {dest} ({len(raw)} bytes)")
    if "seed" in output:
        print(f"  seed:  {output['seed']}")
    if "loras" in output:
        print(f"  loras: {output['loras']}")
    if "width" in output and "height" in output:
        print(f"  size:  {output['width']}x{output['height']}")
    delay = result.get("delayTime")
    exec_t = result.get("executionTime")
    if delay is not None or exec_t is not None:
        print(f"  times: delay={delay}ms exec={exec_t}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
