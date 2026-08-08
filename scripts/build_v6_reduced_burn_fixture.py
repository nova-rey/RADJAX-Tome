#!/usr/bin/env python3
"""Build the published ordinary-production P6.U1 reduced-burn artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from radjax_tome.tome.v6_reduced_burn_fixture import (
    build_v6_reduced_burn_fixture,
    build_v6_reduced_burn_pair,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compare-output",
        type=Path,
        help="also build a fresh comparison output and attach pair evidence",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.compare_output is None:
        build_v6_reduced_burn_fixture(
            args.output, spec_path=args.spec, overwrite=args.overwrite
        )
    else:
        if args.overwrite:
            raise SystemExit("--overwrite is not supported with --compare-output")
        build_v6_reduced_burn_pair(
            args.output, args.compare_output, spec_path=args.spec
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
