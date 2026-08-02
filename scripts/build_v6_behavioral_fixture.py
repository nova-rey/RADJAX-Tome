#!/usr/bin/env python3
"""Build the checked-in ordinary-production native-v3 Student v6 fixture."""

from __future__ import annotations

import argparse

from radjax_tome.tome.v6_fixture import build_v6_behavioral_fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_v6_behavioral_fixture(args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
