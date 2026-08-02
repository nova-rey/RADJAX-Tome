#!/usr/bin/env python3
"""Build the checked-in native v5 smoke-tokenizer evidence fixture."""

from __future__ import annotations

import argparse

from radjax_tome.tome.v5_fixture import build_v5_language_tokenizer_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tome-commit", default="3861c23")
    args = parser.parse_args()
    build_v5_language_tokenizer_fixture(
        args.output,
        overwrite=args.overwrite,
        tome_commit=args.tome_commit,
    )


if __name__ == "__main__":
    main()
