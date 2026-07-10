from __future__ import annotations

import argparse
from pathlib import Path

from radjax_tome.tome.golden_fixture import build_production_contract_fixture


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic production Tome Contract fixture."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--producer-commit", default=None)
    args = parser.parse_args()
    kwargs = {"overwrite": args.overwrite}
    if args.producer_commit is not None:
        kwargs["producer_commit"] = args.producer_commit
    artifact = build_production_contract_fixture(args.output, **kwargs)
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
