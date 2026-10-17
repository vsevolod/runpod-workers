"""Request contract for image_generate / image_edit job inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[2]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.api import CommonRequest, KREA_API, RequestError, normalize_common_fields
from shared.api.images import decode_image_string


@dataclass(frozen=True)
class NormalizedRequest(CommonRequest):
    width: int | None
    height: int | None
    size_from_source: bool
    mu: float | None
    grounding_px: int
    ref_boost: float
    fit_mode: str


def normalize_job_input(
    validated: dict,
    *,
    raw_keys: set[str] | None = None,
) -> NormalizedRequest:
    """
    validated: parsed shared API fields with defaults and compatibility aliases.
    raw_keys: explicit client fields before defaults — used for edit size.
    """
    raw_keys = set(raw_keys or ())

    common = normalize_common_fields(validated, KREA_API, raw_keys=raw_keys)

    if common.type == "image_generate":
        size_from_source = False
        width = int(validated["width"])
        height = int(validated["height"])
        if width % 16 != 0 or height % 16 != 0:
            raise RequestError(
                f"width and height must be multiples of 16 (got {width}x{height})"
            )
    else:
        size_from_source = ("width" not in raw_keys) and ("height" not in raw_keys)
        if size_from_source:
            # Canvas primary is images[0]; both None (handler/pipeline invariant).
            width, height = None, None
        else:
            width = int(validated["width"])
            height = int(validated["height"])
            if width % 16 != 0 or height % 16 != 0:
                raise RequestError(
                    f"width and height must be multiples of 16 (got {width}x{height})"
                )

    fit_mode = validated.get("fit_mode") or "fit"
    if fit_mode not in ("fit", "crop"):
        raise RequestError("fit_mode must be 'fit' or 'crop'")

    grounding_px = int(
        validated.get("grounding_px") if validated.get("grounding_px") is not None else 768
    )
    if grounding_px != 0 and not (256 <= grounding_px <= 2048):
        raise RequestError("grounding_px must be 0 or in 256..2048")

    ref_boost = float(
        validated.get("ref_boost") if validated.get("ref_boost") is not None else 1.0
    )
    if not math.isfinite(ref_boost) or ref_boost < 0:
        raise RequestError("ref_boost must be a finite number >= 0")

    return NormalizedRequest(
        **vars(common), width=width, height=height, size_from_source=size_from_source,
        mu=validated.get("mu"), grounding_px=grounding_px,
        ref_boost=ref_boost, fit_mode=fit_mode,
    )
