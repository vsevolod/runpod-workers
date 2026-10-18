"""Request contract for image_generate / image_edit job inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[2]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.api import CommonRequest, KREA_API, RequestError, normalize_common_fields
from shared.api.images import decode_image_string
from shared.megapixels import canvas_at_megapixels, resize_references


@dataclass(frozen=True)
class NormalizedRequest(CommonRequest):
    width: int
    height: int
    size_from_source: bool
    mu: float | None
    grounding_px: int
    ref_boost: float
    ref_boost_a: float
    ref_position: str
    ref_position_a: str
    fit_mode: str


def normalize_job_input(
    validated: dict,
    *,
    raw_keys: set[str] | None = None,
    quant_mode: str | None = None,
) -> NormalizedRequest:
    """
    validated: parsed shared API fields with defaults and compatibility aliases.
    raw_keys: explicit client fields before defaults — used for edit size.
    quant_mode: canonical mode of the resident pipeline, used for the VRAM table.
    """
    raw_keys = set(raw_keys or ())

    common = normalize_common_fields(validated, KREA_API, raw_keys=raw_keys,
                                     megapixel_model=f"krea2.{quant_mode or 'fp8'}")

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
            width, height = canvas_at_megapixels(common.images[0], common.reference_megapixels, multiple=16)
            if not (256 <= width <= KREA_API.max_side and 256 <= height <= KREA_API.max_side):
                raise RequestError("automatic canvas is outside 256..2048; pass explicit width and height")
        else:
            width = int(validated["width"])
            height = int(validated["height"])
            if width % 16 != 0 or height % 16 != 0:
                raise RequestError(
                    f"width and height must be multiples of 16 (got {width}x{height})"
                )

    try:
        common = replace(common, images=resize_references(common.images, common.reference_megapixels,
                                                        multiple=1, max_side=KREA_API.max_side))
    except ValueError as err:
        raise RequestError(str(err)) from err

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

    ref_boost_a = float(validated.get("ref_boost_a", 1.))
    if not math.isfinite(ref_boost_a) or ref_boost_a < 0:
        raise RequestError("ref_boost_a must be a finite number >= 0")
    ref_position = validated.get("ref_position", "aligned")
    ref_position_a = validated.get("ref_position_a", "aligned")
    if ref_position not in ("aligned", "beside", "bag_of_features") or ref_position_a not in ("aligned", "beside", "bag_of_features"):
        raise RequestError("reference positions must be aligned, beside or bag_of_features")

    return NormalizedRequest(
        **vars(common), width=width, height=height, size_from_source=size_from_source,
        mu=validated.get("mu"), grounding_px=grounding_px,
        ref_boost=ref_boost, fit_mode=fit_mode,
        ref_boost_a=ref_boost_a, ref_position=ref_position, ref_position_a=ref_position_a,
    )
