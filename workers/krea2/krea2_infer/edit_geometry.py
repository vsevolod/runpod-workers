"""Source pixel geometry for identity edit (fit / crop)."""

from __future__ import annotations

from PIL import Image as PILImage


def target_size_from_source(
    width: int,
    height: int,
    *,
    max_megapixels: float = 1.5,
    align: int = 16,
) -> tuple[int, int]:
    """Return (width, height) both multiples of align, AR-preserved, MP-capped."""
    w, h = float(width), float(height)
    if w <= 0 or h <= 0:
        raise ValueError(f"width and height must be positive (got {width}x{height})")
    ar = w / h
    mp = (w * h) / 1e6
    if max_megapixels > 0 and mp > max_megapixels:
        s = (max_megapixels / mp) ** 0.5
        w, h = w * s, h * s
    # Snap width, then re-derive height from AR so aspect stays close after align.
    w = max(align, int(round(w / align)) * align)
    h = max(align, int(round((w / ar) / align)) * align)
    # If MP still over after round-up, floor-snap both.
    if max_megapixels > 0 and (w * h) / 1e6 > max_megapixels + 1e-9:
        w = max(align, int(w) // align * align)
        h = max(align, int(round((w / ar) / align)) * align)
        while (w * h) / 1e6 > max_megapixels and (w > align or h > align):
            if w >= h and w > align:
                w = max(align, w - align)
            elif h > align:
                h = max(align, h - align)
            else:
                break
            h = max(align, int(round((w / ar) / align)) * align)
    return int(w), int(h)


def fit_source_pixels(
    image: PILImage.Image,
    target_h: int,
    target_w: int,
    fit_mode: str = "fit",
) -> PILImage.Image:
    """
    Pixel-space prep before VAE encode.
    - crop: center-crop to target AR, bicubic to exactly (target_w, target_h).
    - fit: AR-preserving fit-inside with /16 floor snap (Comfy _fit_encode_image);
      may return size <= target; never non-uniform stretch.
    Returns RGB PIL.Image with size (W, H) = (pixel_w, pixel_h).
    """
    img = image.convert("RGB")
    iw, ih = img.size
    px_h, px_w = target_h, target_w  # already pixel space

    if fit_mode == "crop" or fit_mode == "crop (legacy)":
        s = max(px_h / ih, px_w / iw)
        ch = min(ih, int(round(px_h / s)))
        cw = min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        return img.resize((px_w, px_h), PILImage.Resampling.BICUBIC)

    if fit_mode != "fit":
        raise ValueError(f"unknown fit_mode: {fit_mode}")

    # fit: Comfy near-match CROP_TOL=0.08 else /16 floor snap fit-inside
    sc = min(px_h / ih, px_w / iw)
    crop_tol = 0.08
    if ih * sc >= px_h * (1 - crop_tol) and iw * sc >= px_w * (1 - crop_tol):
        s = max(px_h / ih, px_w / iw)
        ch = min(ih, int(round(px_h / s)))
        cw = min(iw, int(round(px_w / s)))
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        return img.resize((px_w, px_h), PILImage.Resampling.BICUBIC)

    nh = min(max(16, int(ih * sc) // 16 * 16), max(16, px_h // 16 * 16))
    nw = min(max(16, int(iw * sc) // 16 * 16), max(16, px_w // 16 * 16))
    return img.resize((nw, nh), PILImage.Resampling.BICUBIC)
