#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from radjax_tome.targets import inspect_target_store


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a teacher target store")
    parser.add_argument("target_store", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary")
    args = parser.parse_args()

    summary = inspect_target_store(args.target_store)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print("Target store summary")
    print("====================")
    for key in (
        "target_type",
        "target_store_version",
        "model_id",
        "tokenizer_id",
        "vocab_size",
        "sequence_length",
        "num_examples",
        "shard_count",
        "array_keys",
    ):
        print(f"{key}: {summary[key]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Invalid target store: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
