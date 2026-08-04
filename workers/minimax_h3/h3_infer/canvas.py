"""Canvas sizing — algorithm parity with diffusers packing.py @ abc5e9bf.

API width/height are pixel W×H. ``resolve_canvas_size`` returns (height, width)
matching upstream.
"""

from __future__ import annotations

MINIMAX_H3_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344  # 1_032_192
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_ASPECT_RATIO = 1 / 4
MINIMAX_H3_MAX_ASPECT_RATIO = 4


def resolve_canvas_size(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve a display aspect ratio into a MiniMax-H3 canvas (height, width).

    Byte-for-algorithm match of
    ``diffusers.modular_pipelines.minimax_h3.packing.resolve_canvas_size``
    @ abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}."
        )

    ratio = aspect_width / aspect_height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, "
            f"got {aspect_width}:{aspect_height} ({ratio:g})."
        )

    if ratio >= 1.0:
        width, height = MINIMAX_H3_SHORT_EDGE * ratio, float(MINIMAX_H3_SHORT_EDGE)
    else:
        width, height = float(MINIMAX_H3_SHORT_EDGE), MINIMAX_H3_SHORT_EDGE / ratio

    area = width * height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = (MINIMAX_H3_MAX_PIXELS / area) ** 0.5
        width, height = width * scale, height * scale

    multiple = MINIMAX_H3_CANVAS_MULTIPLE
    return (
        max(multiple, round(height / multiple) * multiple),
        max(multiple, round(width / multiple) * multiple),
    )


def validate_canvas(width: int, height: int) -> None:
    """Validate client pixel size against packing nominal for the same aspect.

    Accepts previews with area ≤ nominal (e.g. 864×480). Accepts official long
    canvases such as 2016×512. Rejects non-multiples of 32 and oversize area.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise ValueError(f"width must be int, got {type(width).__name__}")
    if not isinstance(height, int) or isinstance(height, bool):
        raise ValueError(f"height must be int, got {type(height).__name__}")
    if width < 32 or height < 32:
        raise ValueError(f"width and height must be >= 32, got {width}x{height}")
    if width % MINIMAX_H3_CANVAS_MULTIPLE != 0 or height % MINIMAX_H3_CANVAS_MULTIPLE != 0:
        raise ValueError(
            f"width and height must be multiples of {MINIMAX_H3_CANVAS_MULTIPLE}, "
            f"got {width}x{height}"
        )

    aspect = width / height
    if not MINIMAX_H3_MIN_ASPECT_RATIO <= aspect <= MINIMAX_H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"aspect ratio width/height must be in "
            f"[{MINIMAX_H3_MIN_ASPECT_RATIO}, {MINIMAX_H3_MAX_ASPECT_RATIO}], "
            f"got {aspect:g} ({width}x{height})"
        )

    # Ratio only: pass client W,H as aspect_width, aspect_height.
    nom_h, nom_w = resolve_canvas_size(width, height)
    if width * height > nom_h * nom_w:
        raise ValueError(
            f"canvas area {width}x{height}={width * height} exceeds nominal "
            f"{nom_w}x{nom_h}={nom_h * nom_w} for this aspect"
        )
