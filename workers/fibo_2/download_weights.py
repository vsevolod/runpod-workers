#!/usr/bin/env python3
"""CPU-only downloader for the selected pinned Fibo 2 base or turbo checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.model_download import add_download_arguments
from shared.prefetch import ComponentPublisher
from fibo_2_infer.config import CHECKPOINTS, model_directory
from fibo_2_infer.model_store import download_model


def main():
    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument('--variant', choices=tuple(CHECKPOINTS), default='base')
    selected, _ = selection.parse_known_args()
    parser = argparse.ArgumentParser(description=__doc__, parents=[selection])
    add_download_arguments(parser, default_output=str(model_directory(selected.variant)))
    args = parser.parse_args()
    download_model(args.output, variant=args.variant, hf_home=args.hf_home, token=args.token,
                   publisher=ComponentPublisher(args.prefetch_state_dir))


if __name__ == '__main__':
    main()
