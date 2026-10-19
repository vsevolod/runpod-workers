#!/usr/bin/env python3
"""Download pinned MOSS-Image-SFT data files, optionally its Turbo adapter."""

import argparse
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from shared.model_download import add_download_arguments
from shared.prefetch import ComponentPublisher
from moss_image_infer.config import model_directory
from moss_image_infer.model_store import download_model, ensure_turbo_lora


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output=str(model_directory()))
    parser.set_defaults(output=str(model_directory()))
    parser.add_argument('--turbo', action='store_true', help='Also prefetch the pinned DMD LoRA')
    args = parser.parse_args(argv)
    root = Path(args.output).expanduser().resolve()
    download_model(root, hf_home=args.hf_home, token=args.token)
    if args.turbo:
        ensure_turbo_lora(root, hf_home=args.hf_home, token=args.token)
    ComponentPublisher(args.prefetch_state_dir).ready('all', root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
