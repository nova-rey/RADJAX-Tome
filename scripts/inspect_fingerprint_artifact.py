#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys

from radjax_tome.fingerprint.artifacts import summarize_fingerprint_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a producer behavioral fingerprint artifact"
    )
    parser.add_argument("artifact", help="Fingerprint artifact directory")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    try:
        summary = summarize_fingerprint_artifact(args.artifact)
    except ValueError as exc:
        print(f"Fingerprint artifact inspection failed: {exc}", file=sys.stderr)
        return 1

    payload = summary.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
