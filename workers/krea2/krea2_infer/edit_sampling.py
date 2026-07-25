"""Identity-edit sampling: dual conditioning (grounded TE + source VAE tokens)."""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from .edit_geometry import fit_source_pixels, target_size_from_source
from .sampling import _module_device, _offload_encoder_to_cpu, roundup, timesteps

logger = logging.getLogger(__name__)


def _img_ids(bs: int, frame: float, gh: int, gw: int, device) -> torch.Tensor:
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(gh, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(gw, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, gh * gw, 3).expand(bs, -1, -1).contiguous()


def _img_ids_offset(bs, frame, gh, gw, th, tw, device) -> torch.Tensor:
    off_h, off_w = max(0, (th - gh) // 2), max(0, (tw - gw) // 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = (torch.arange(gh, device=device, dtype=torch.float32) + off_h)[:, None]
    ids[..., 2] = (torch.arange(gw, device=device, dtype=torch.float32) + off_w)[None, :]
    return ids.reshape(1, gh * gw, 3).expand(bs, -1, -1).contiguous()


def build_edit_position_ids(
    batch: int,
    text_len: int,
    src_grids: list[tuple[int, int]],
    tgt_grid: tuple[int, int],
    *,
    pos_mode: str,
    device,
) -> torch.Tensor:
    """Returns (B, text_len + sum(gh*gw) + th*tw, 3)."""
    th, tw = tgt_grid
    txt = torch.zeros(batch, text_len, 3, device=device, dtype=torch.float32)
    blocks = [txt]
    for i, (gh, gw) in enumerate(src_grids):
        frame = float(i + 1)
        if pos_mode == "stride1":
            blocks.append(_img_ids_offset(batch, frame, gh, gw, th, tw, device))
        else:
            blocks.append(_img_ids(batch, frame, gh, gw, device))
    blocks.append(_img_ids(batch, 0.0, th, tw, device))
    return torch.cat(blocks, dim=1)


def build_ref_boost_bias(
    boosts: list[float],
    text_len: int,
    src_token_lens: list[int],
    tgt_len: int,
    *,
    device,
    dtype,
) -> torch.Tensor | None:
    """(1,1,L,L) additive logits or None if all boosts == 1.0."""
    if all(b == 1.0 for b in boosts):
        return None
    L = text_len + sum(src_token_lens) + tgt_len
    bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
    off = text_len
    rows0 = text_len + sum(src_token_lens)
    for b, sl in zip(boosts, src_token_lens):
        if b != 1.0:
            bias[:, :, rows0:, off : off + sl] = math.log(max(float(b), 1e-4))
        off += sl
    return bias


@torch.no_grad()
def sample_edit(
    model,  # SingleStreamDiT
    ae,  # QwenAutoencoder
    encoder,  # Qwen3VLConditioner
    instruction: str,
    source: Image.Image,
    *,
    negative_prompt: str | None = None,
    device="cuda",
    dtype=torch.bfloat16,
    width: int | None = None,
    height: int | None = None,
    steps: int = 8,
    guidance: float = 0.0,
    seed: int = 0,
    mu: float | None = 1.15,
    grounding_px: int = 768,
    ref_boost: float = 1.0,
    fit_mode: str = "fit",
    max_megapixels: float = 1.5,
    lora_activation=None,
) -> list[Image.Image]:
    patch = model.config.patch
    align = ae.compression * patch  # 16

    sw, sh = source.size
    if width is None or height is None:
        width, height = target_size_from_source(
            sw, sh, max_megapixels=max_megapixels, align=align
        )
    else:
        width, height = roundup(width, align, "width"), roundup(height, align, "height")

    # --- TE on GPU ---
    if _module_device(encoder).type != "cuda" and str(device).startswith("cuda"):
        encoder.to(device)

    txt, txtmask = encoder.grounded_encode(
        [instruction], [source], grounding_px=grounding_px
    )
    # txt: (1, T, 12, C), txtmask: (1, T)
    cfg = guidance > 0
    if cfg:
        neg_text = negative_prompt if negative_prompt is not None else ""
        untxt, untxtmask = encoder.grounded_encode(
            [neg_text], [source], grounding_px=grounding_px
        )

    _offload_encoder_to_cpu(encoder)

    # --- source pixels → latent tokens ---
    src_img = fit_source_pixels(source, target_h=height, target_w=width, fit_mode=fit_mode)
    arr = np.asarray(src_img, dtype=np.float32) / 255.0  # H,W,3
    px = torch.from_numpy(arr).to(device=device, dtype=dtype)
    px = px.permute(2, 0, 1).unsqueeze(0) * 2 - 1  # (1,3,hs,ws)
    src_lat = ae.encode(px)  # (1,16,hs/8,ws/8)
    src_tok = rearrange(
        src_lat, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
    )
    src_len = src_tok.shape[1]
    src_gh, src_gw = src_lat.shape[-2] // patch, src_lat.shape[-1] // patch

    # --- target noise ---
    lh, lw = height // ae.compression, width // ae.compression
    noise = torch.randn(
        1,
        ae.channels,
        lh,
        lw,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    tgt_tok = rearrange(
        noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch
    )
    tgt_len = tgt_tok.shape[1]
    th, tw = lh // patch, lw // patch

    pos_mode = "stride1" if fit_mode == "fit" else "anchor"
    pos = build_edit_position_ids(
        1,
        txt.shape[1],
        [(src_gh, src_gw)],
        (th, tw),
        pos_mode=pos_mode,
        device=device,
    )
    img_ones = torch.ones(1, src_len + tgt_len, device=device, dtype=torch.bool)
    mask = torch.cat([txtmask.to(device), img_ones], dim=1)

    bias = build_ref_boost_bias(
        [ref_boost],
        txt.shape[1],
        [src_len],
        tgt_len,
        device=device,
        dtype=dtype,
    )

    combined = torch.cat([src_tok, tgt_tok], dim=1)  # (1, S+G, Cpack)

    # Timestep schedule: pin mu (default 1.15). Never use resolution-auto mu with S+G.
    if mu is None:
        mu = 1.15
    x1 = (256 // align) ** 2
    x2 = (1280 // align) ** 2
    ts = timesteps(tgt_len, steps, x1, x2, mu=mu)

    img = combined
    activation_scope = nullcontext() if lora_activation is None else lora_activation
    with activation_scope:
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=img.device)
            cond = model(
                img=img, context=txt, t=t, pos=pos, mask=mask, attn_bias=bias
            )
            # cond: (1, S+G, Cpack) — keep target only
            cond = cond[:, -tgt_len:, :]
            if cfg:
                unpos = build_edit_position_ids(
                    1,
                    untxt.shape[1],
                    [(src_gh, src_gw)],
                    (th, tw),
                    pos_mode=pos_mode,
                    device=device,
                )
                unmask = torch.cat([untxtmask.to(device), img_ones], dim=1)
                unbias = build_ref_boost_bias(
                    [ref_boost],
                    untxt.shape[1],
                    [src_len],
                    tgt_len,
                    device=device,
                    dtype=dtype,
                )
                uncond = model(
                    img=img,
                    context=untxt,
                    t=t,
                    pos=unpos,
                    mask=unmask,
                    attn_bias=unbias,
                )[:, -tgt_len:, :]
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            # Euler only on target tokens; source tokens stay clean
            tgt = img[:, -tgt_len:, :] + (tprev - tcurr) * v
            img = torch.cat([img[:, :src_len, :], tgt], dim=1)

    tgt = img[:, -tgt_len:, :]
    lat = rearrange(
        tgt,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=th,
        w=tw,
    )
    pixels = ae.decode(lat.to(torch.bfloat16))
    pixels = pixels.clamp(-1, 1) * 0.5 + 0.5
    pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(pixels[0])]
