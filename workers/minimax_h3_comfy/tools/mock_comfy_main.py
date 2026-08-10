#!/usr/bin/env python3
"""Tiny stand-in for ComfyUI main.py — only /system_stats for boot smoke."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # quieter
        print(f"mock_comfy: {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in ("/system_stats", "/"):
            body = json.dumps(
                {"system": {"os": "mock"}, "devices": [{"name": "mock-gpu"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--listen", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    # Accept and ignore real Comfy flags
    p.add_argument("--disable-auto-launch", action="store_true")
    p.add_argument("--disable-metadata", action="store_true")
    p.add_argument("--log-stdout", action="store_true")
    p.add_argument("--verbose", default=None)
    args, _unknown = p.parse_known_args()

    host = args.listen if args.listen not in ("0.0.0.0", "") else "0.0.0.0"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"mock_comfy: listening on http://{host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
