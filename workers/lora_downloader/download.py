from __future__ import annotations

from pathlib import Path

SAFETENSORS_SUFFIX = ".safetensors"


class FilenameError(ValueError):
    """Invalid LoRA filename (safe client message)."""


def normalize_filename(name: object) -> str:
    if not isinstance(name, str):
        raise FilenameError("filename must be a string")
    name = name.strip()
    if not name:
        raise FilenameError("filename is empty")
    if "/" in name or "\\" in name:
        raise FilenameError("filename must not contain path separators")
    for ch in name:
        o = ord(ch)
        if o < 32 or o == 127:
            raise FilenameError("filename contains control characters")
    if not name.endswith(SAFETENSORS_SUFFIX):
        raise FilenameError("filename must end with .safetensors")
    stem = name[: -len(SAFETENSORS_SUFFIX)]
    if not stem or stem in (".", ".."):
        raise FilenameError("filename stem is invalid")
    return name
