#!/usr/bin/env python3
"""Download pinned MOSS-Image-SFT data files and the selected built-in adapter."""

import argparse
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from shared.model_download import add_download_arguments
from shared.prefetch import ComponentPublisher
from moss_image_infer.config import model_directory
from moss_image_infer.model_store import download_model, ensure_turbo_lora, ensure_edit_lora


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output=str(model_directory()))
    parser.set_defaults(output=str(model_directory()))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--turbo', action='store_true', help='Also prefetch the pinned DMD LoRA')
    selection.add_argument('--edit', action='store_true', help='Also prefetch the pinned Edit LoRA/config')
    parser.add_argument('--lora-only', action='store_true', help='Prepare the selected adapter on resident SFT (default: Turbo)')
    args = parser.parse_args(argv)
    root = Path(args.output).expanduser().resolve()
    publisher = ComponentPublisher(args.prefetch_state_dir)
    if not args.lora_only:
        download_model(root, hf_home=args.hf_home, token=args.token, publisher=publisher)
    if args.edit:
        adapter = ensure_edit_lora(root, hf_home=args.hf_home, token=args.token)
        publisher.ready('edit_lora', adapter)
    elif args.turbo or args.lora_only:
        adapter = ensure_turbo_lora(root, hf_home=args.hf_home, token=args.token)
        publisher.ready('turbo_lora', adapter)
    publisher.ready('all', root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
