#!/usr/bin/env python3
"""Compatibility command for Contract-owned M7 portable validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radjax_contract.tome import validate_streaming_tome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = validate_streaming_tome(args.path)
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
