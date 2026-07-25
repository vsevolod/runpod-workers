#!/usr/bin/env python3
"""RunPod Krea2 client: text2img, image_edit, or fetch by Request ID → PNG.

Modes (auto-detected):
  1) Fetch:     --request-id <id>  → status (poll if needed) → PNG
  2) Edit:      -i/--image <path>  → image_edit job → PNG
  3) Generate:  otherwise          → text2img job → PNG

Env:
  ENDPOINT_ID      RunPod serverless endpoint id
  RUNPOD_API_KEY   API key (Bearer)

Examples:
  # Text-to-image (interactive prompt/LoRA if omitted)
  python krea2_image.py -o out.png
  python krea2_image.py --prompt "a red fox" --lora my_lora@0.8 -o out.png

  # Image edit
  python krea2_image.py -i person.jpg -o edit.png \\
    --prompt "change the jacket to red leather" \\
    --lora krea2_identity_edit_v1_2 --ref-boost 4 --seed 42

  # Download an already submitted job
  python krea2_image.py --request-id abc123 -o out.png
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
POLL_INTERVAL_S = 2.0
DEFAULT_OUTPUT = "output.png"
DEFAULT_EDIT_OUTPUT = "edit_output.png"
DEFAULT_EDIT_LORA = "krea2_identity_edit_v1_2"
DEFAULT_REF_BOOST = 4.0
DEFAULT_GROUNDING_PX = 768
DEFAULT_FIT_MODE = "fit"
DEFAULT_STEPS = 8
DEFAULT_GEN_SIZE = 1024


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
    """Avoid dumping multi-MB base64 into the terminal."""
    out = dict(job_input)
    images = out.get("images")
    if isinstance(images, list):
        redacted = []
        for item in images:
            if isinstance(item, str) and len(item) > 80:
                redacted.append(f"<{len(item)} chars data URL/base64>")
            else:
                redacted.append(item)
        out["images"] = redacted
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


def _maybe_save_json(result: dict[str, Any], output: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    dump = dict(result)
    out_copy = dict(output)
    if isinstance(out_copy.get("images"), list):
        out_copy["images"] = [
            f"<{len(x)} chars>" if isinstance(x, str) else x
            for x in out_copy["images"]
        ]
    if isinstance(out_copy.get("image_url"), str):
        out_copy["image_url"] = f"<{len(out_copy['image_url'])} chars>"
    dump["output"] = out_copy
    Path(path).write_text(
        json.dumps(dump, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  meta:   {path}")


def _save_result(
    result: dict[str, Any],
    dest: Path,
    *,
    save_json: str | None = None,
) -> int:
    """Decode job result and write PNG. Returns process exit code."""
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
        "ref_boost",
        "fit_mode",
        "grounding_px",
        "loras",
    ):
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
        "-o",
        "--output",
        default=None,
        help=(
            f"Output PNG path (default: {DEFAULT_OUTPUT} for generate/fetch, "
            f"{DEFAULT_EDIT_OUTPUT} for edit)"
        ),
    )
    parser.add_argument(
        "--request-id",
        "--job-id",
        dest="request_id",
        default=None,
        help=(
            "Existing RunPod job/request id: skip submit, fetch status "
            "(poll until done if still running), save image."
        ),
    )
    parser.add_argument(
        "-i",
        "--image",
        default=None,
        help="Source image path → image_edit mode (PNG/JPEG).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt / edit instruction. If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help=(
            "LoRA catalog id(s): name or name@strength, comma-separated. "
            f"Edit default: {DEFAULT_EDIT_LORA}. Generate: interactive if omitted. "
            "Empty string = no LoRA."
        ),
    )
    parser.add_argument(
        "--ref-boost",
        type=float,
        default=DEFAULT_REF_BOOST,
        help=f"Edit only: ref_boost (default: {DEFAULT_REF_BOOST})",
    )
    parser.add_argument(
        "--grounding-px",
        type=int,
        default=DEFAULT_GROUNDING_PX,
        help=f"Edit only: VLM long-side cap; 0 = native (default: {DEFAULT_GROUNDING_PX})",
    )
    parser.add_argument(
        "--fit-mode",
        choices=("fit", "crop"),
        default=DEFAULT_FIT_MODE,
        help=f"Edit only: source geometry fit | crop (default: {DEFAULT_FIT_MODE})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            f"Target width. Generate default: {DEFAULT_GEN_SIZE}. "
            "Edit: omit both W/H to derive from source."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            f"Target height. Generate default: {DEFAULT_GEN_SIZE}. "
            "Edit: omit both W/H to derive from source."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed")
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"num_inference_steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="CFG (default: 0.0)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1.15,
        help="Turbo shift mu (default: 1.15)",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt (used when guidance_scale > 0)",
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
        help="Optional path to write full job status JSON (images redacted)",
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
    print(f"mode:     fetch")
    print(f"endpoint: {endpoint_id}")
    print(f"request:  {job_id}")
    print()
    result = _poll_job(endpoint_id, api_key, job_id, interval=interval)
    print()
    return _save_result(result, output, save_json=save_json)


def _run_edit(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    if (args.width is None) ^ (args.height is None):
        print(
            "error: pass both --width and --height, or neither (size from source)",
            file=sys.stderr,
        )
        return 1

    image_path = Path(args.image).expanduser()
    data_url = _image_to_data_url(image_path)

    prompt = args.prompt
    if prompt is None:
        print(f"mode:     edit")
        print(f"endpoint: {endpoint_id}")
        print(f"source:   {image_path} ({image_path.stat().st_size} bytes)")
        print()
        prompt = _prompt_line("prompt (edit instruction)")
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 1

    lora_raw = args.lora if args.lora is not None else DEFAULT_EDIT_LORA
    loras = _parse_lora(lora_raw)

    job_input: dict[str, Any] = {
        "type": "image_edit",
        "prompt": prompt,
        "images": [data_url],
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "mu": args.mu,
        "grounding_px": args.grounding_px,
        "ref_boost": args.ref_boost,
        "fit_mode": args.fit_mode,
        "loras": loras,
    }
    if args.width is not None and args.height is not None:
        job_input["width"] = args.width
        job_input["height"] = args.height
    if args.seed is not None:
        job_input["seed"] = args.seed
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt

    out = Path(args.output or DEFAULT_EDIT_OUTPUT)
    print(f"mode:     edit")
    print(f"endpoint: {endpoint_id}")
    print(f"source:   {image_path} ({image_path.stat().st_size} bytes → data URL)")
    print(f"prompt:   {prompt}")
    print(f"loras:    {loras or '(none)'}")
    print(
        f"ref_boost={args.ref_boost}  fit_mode={args.fit_mode}  "
        f"grounding_px={args.grounding_px}  steps={args.steps}"
    )
    if args.width is not None:
        print(f"size:     {args.width}x{args.height} (explicit)")
    else:
        print("size:     from source (omit width/height)")
    print()

    job_id = _submit_job(endpoint_id, api_key, job_input)
    print()
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    print()
    return _save_result(result, out, save_json=args.save_json)


def _run_generate(args: argparse.Namespace, *, endpoint_id: str, api_key: str) -> int:
    width = args.width if args.width is not None else DEFAULT_GEN_SIZE
    height = args.height if args.height is not None else DEFAULT_GEN_SIZE

    print(f"mode:     generate")
    print(f"endpoint: {endpoint_id}")
    print()

    prompt = args.prompt
    if prompt is None:
        prompt = _prompt_line("prompt")
    if not prompt:
        print("error: prompt is required", file=sys.stderr)
        return 1

    if args.lora is not None:
        lora_raw = args.lora
    else:
        lora_raw = _prompt_line(
            "lora (name or name@strength, comma-separated; empty = none)",
            default="",
        )
    loras = _parse_lora(lora_raw)

    job_input: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "mu": args.mu,
        "loras": loras,
    }
    if args.seed is not None:
        job_input["seed"] = args.seed
    if args.negative_prompt:
        job_input["negative_prompt"] = args.negative_prompt

    out = Path(args.output or DEFAULT_OUTPUT)
    print()
    print(f"prompt:   {prompt}")
    print(f"loras:    {loras or '(none)'}")
    print(f"size:     {width}x{height}  steps={args.steps}")
    print()

    job_id = _submit_job(endpoint_id, api_key, job_input)
    print()
    result = _poll_job(
        endpoint_id, api_key, job_id, interval=max(0.5, args.poll_interval)
    )
    print()
    return _save_result(result, out, save_json=args.save_json)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    endpoint_id = _require_env("ENDPOINT_ID")
    api_key = _require_env("RUNPOD_API_KEY")
    interval = max(0.5, args.poll_interval)

    if args.request_id:
        if args.image:
            print(
                "error: --request-id cannot be combined with -i/--image "
                "(fetch does not submit a new job)",
                file=sys.stderr,
            )
            return 1
        out = Path(args.output or DEFAULT_OUTPUT)
        return _run_fetch(
            endpoint_id=endpoint_id,
            api_key=api_key,
            request_id=args.request_id,
            output=out,
            interval=interval,
            save_json=args.save_json,
        )

    if args.image:
        return _run_edit(args, endpoint_id=endpoint_id, api_key=api_key)

    return _run_generate(args, endpoint_id=endpoint_id, api_key=api_key)


if __name__ == "__main__":
    raise SystemExit(main())
