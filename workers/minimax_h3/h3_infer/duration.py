"""Frame snapping and duration bounds for MiniMax H3 T2V (24 fps, 17n+5)."""

from __future__ import annotations

FPS = 24
MIN_DURATION_SEC = 5.0
# Last 17n+5 frame count with output_duration <= 15 s (345 / 24).
MAX_DURATION_SEC = 14.375


def snap_num_frames(duration_sec: float) -> int:
    """Map duration to nearest 17n+5 frame count (VAE packing contract)."""
    frames = max(5, round(float(duration_sec) * FPS))
    return frames + (5 - (frames % 17)) % 17


def validate_requested_duration(duration_sec: float) -> float:
    """Validate client duration is in [MIN_DURATION_SEC, MAX_DURATION_SEC]."""
    try:
        d = float(duration_sec)
    except (TypeError, ValueError) as err:
        raise ValueError(f"duration must be a number, got {duration_sec!r}") from err
    if d < MIN_DURATION_SEC or d > MAX_DURATION_SEC:
        raise ValueError(
            f"duration must be in [{MIN_DURATION_SEC}, {MAX_DURATION_SEC}] seconds, got {d}"
        )
    return d


def output_duration_sec(num_frames: int) -> float:
    """Wall-clock seconds of snapped frame count at FPS."""
    return float(num_frames) / float(FPS)
