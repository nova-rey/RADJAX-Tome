#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from radjax_tome.fingerprint import validate_fingerprint_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a producer behavioral fingerprint artifact"
    )
    parser.add_argument("artifact", help="Fingerprint artifact directory")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    result = validate_fingerprint_artifact(args.artifact)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"status: {result.status}")
        for blocker in result.blockers:
            print(f"blocker: {blocker}")
        for warning in result.warnings:
            print(f"warning: {warning}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
