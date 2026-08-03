#!/usr/bin/env python3
"""Download CivitAI LoRAs via the lora_downloader RunPod endpoint.

Env:
  RUNPOD_API_KEY   required
  ENDPOINT_ID      required unless --endpoint-id is passed

Examples:
  export RUNPOD_API_KEY=rp_xxx
  export ENDPOINT_ID=abc123
  python scripts/download_lora.py 46846
  python scripts/download_lora.py 46846 999 --filename 46846=my.safetensors
  python scripts/download_lora.py 46846 --sfw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any

API_BASE = "https://api.runpod.ai/v2"
POLL_INTERVAL_S = 2.0


def exit_code_for_job_result(result: dict[str, Any]) -> int:
    if result.get("status") != "COMPLETED":
        return 1
    output = result.get("output")
    if not isinstance(output, dict):
        return 1
    if output.get("error"):
        return 1
    summary = output.get("summary") or {}
    try:
        failed = int(summary.get("failed") or 0)
    except (TypeError, ValueError):
        return 1
    if failed > 0:
        return 1
    return 0


def build_job_input(
    version_ids: list[str],
    *,
    filenames: dict[str, str],
    sfw: bool,
) -> dict[str, Any]:
    items = []
    for vid in version_ids:
        item: dict[str, Any] = {
            "model_version_id": vid,
            "nsfw": not sfw,
        }
        if vid in filenames:
            item["filename"] = filenames[vid]
        items.append(item)
    return {"items": items}


def parse_filename_args(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(f"error: --filename must be ID=name.safetensors, got {raw!r}")
        vid, name = raw.split("=", 1)
        vid = vid.strip()
        name = name.strip()
        if not vid or not name:
            raise SystemExit(f"error: bad --filename {raw!r}")
        out[vid] = name
    return out


def require_api_key() -> str:
    value = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not value:
        print("error: set RUNPOD_API_KEY", file=sys.stderr)
        raise SystemExit(1)
    return value


def require_endpoint_id(cli_value: str | None) -> str:
    value = (cli_value or os.environ.get("ENDPOINT_ID") or "").strip()
    if not value:
        print(
            "error: set ENDPOINT_ID env or pass --endpoint-id",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


def api_request(
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
        print(f"error: HTTP {err.code}: {detail[:500]}", file=sys.stderr)
        raise SystemExit(1) from err
    except urllib.error.URLError as err:
        print(f"error: request failed: {err.reason}", file=sys.stderr)
        raise SystemExit(1) from err
    if not raw:
        return {}
    return json.loads(raw)


def submit_job(endpoint_id: str, api_key: str, job_input: dict[str, Any]) -> str:
    url = f"{API_BASE}/{endpoint_id}/run"
    resp = api_request("POST", url, api_key, {"input": job_input}, timeout=120.0)
    job_id = resp.get("id")
    if not job_id:
        print(f"error: no job id: {resp}", file=sys.stderr)
        raise SystemExit(1)
    print(f"job id: {job_id}", file=sys.stderr)
    return str(job_id)


def poll_job(
    endpoint_id: str, api_key: str, job_id: str, *, interval: float
) -> dict[str, Any]:
    url = f"{API_BASE}/{endpoint_id}/status/{job_id}"
    while True:
        resp = api_request("GET", url, api_key, timeout=30.0)
        status = resp.get("status") or "?"
        print(f"status={status}", file=sys.stderr, flush=True)
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return resp
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version_ids",
        nargs="+",
        help="CivitAI model version id(s)",
    )
    parser.add_argument(
        "--filename",
        action="append",
        default=[],
        help="Map version id to filename: ID=name.safetensors (repeatable)",
    )
    parser.add_argument(
        "--sfw",
        action="store_true",
        help="Use civitai.com (nsfw=false) for all items",
    )
    parser.add_argument(
        "--endpoint-id",
        default=None,
        help="RunPod endpoint id (else ENDPOINT_ID env)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL_S,
    )
    args = parser.parse_args(argv)

    api_key = require_api_key()
    endpoint_id = require_endpoint_id(args.endpoint_id)
    filenames = parse_filename_args(args.filename)
    job_input = build_job_input(
        args.version_ids, filenames=filenames, sfw=args.sfw
    )
    job_id = submit_job(endpoint_id, api_key, job_input)
    result = poll_job(
        endpoint_id, api_key, job_id, interval=args.poll_interval
    )
    code = exit_code_for_job_result(result)
    output = result.get("output")
    if isinstance(output, dict):
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if code == 0:
            code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
