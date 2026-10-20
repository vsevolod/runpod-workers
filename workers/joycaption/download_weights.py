#!/usr/bin/env python3
"""CPU-only downloader for the pinned Gliese caption model; no Torch imports."""

import argparse
from pathlib import Path
import sys

workers = Path(__file__).resolve().parents[1]
if (workers / 'shared').is_dir():
    sys.path.insert(0, str(workers))

from shared.model_download import add_download_arguments
from shared.prefetch import ComponentPublisher
from model_store import DEFAULT_MODEL_DIR, download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_download_arguments(parser, default_output=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    download_model(args.output, hf_home=args.hf_home, token=args.token,
                   publisher=ComponentPublisher(args.prefetch_state_dir))


if __name__ == '__main__':
    main()
