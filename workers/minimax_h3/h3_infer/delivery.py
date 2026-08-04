"""Bucket env completeness and delivery mode (url / base64 / error)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

# ALL four required for "configured" (runpod 1.7.9 get_boto_client + upload_file_to_bucket)
BUCKET_ENV_KEYS = (
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
)

MAX_INLINE_VIDEO_BYTES_DEFAULT = 7_000_000


@dataclass(frozen=True)
class DeliveryPlan:
    mode: str  # "url" | "base64" | "error"
    error: str | None = None


def bucket_configured(env: Mapping[str, str] | None = None) -> bool:
    """True when all four bucket credentials are non-empty."""
    e = os.environ if env is None else env
    return all(e.get(k) for k in BUCKET_ENV_KEYS)


def require_bucket_or_exit() -> None:
    """If REQUIRE_BUCKET is truthy, exit when any of the four env vars is missing."""
    if os.environ.get("REQUIRE_BUCKET", "").lower() in {"1", "true", "yes"}:
        missing = [k for k in BUCKET_ENV_KEYS if not os.environ.get(k)]
        if missing:
            raise SystemExit(f"REQUIRE_BUCKET=1 but missing: {', '.join(missing)}")


def choose_delivery(
    has_bucket: bool,
    raw_size: int,
    max_inline: int = MAX_INLINE_VIDEO_BYTES_DEFAULT,
) -> DeliveryPlan:
    """Pick delivery mode from bucket availability and file size (st_size)."""
    if has_bucket:
        return DeliveryPlan(mode="url")
    if raw_size <= max_inline:
        return DeliveryPlan(mode="base64")
    return DeliveryPlan(
        mode="error",
        error=(
            f"video is {raw_size} bytes which exceeds inline limit {max_inline}; "
            "configure all four BUCKET_* env vars for URL delivery"
        ),
    )
