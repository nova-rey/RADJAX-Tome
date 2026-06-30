#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys

from radjax_tome.backends import resolve_qwen_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a local Qwen policy label")
    parser.add_argument("label", help="Qwen policy label, for example Qwen3.latest")
    parser.add_argument(
        "--policy",
        default="configs/qwen_policy.yaml",
        help="Path to local Qwen policy YAML",
    )
    parser.add_argument("--allow-unresolved", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print resolution as JSON")
    args = parser.parse_args()

    try:
        resolution = resolve_qwen_policy(
            args.label,
            policy_path=args.policy,
            allow_unresolved=args.allow_unresolved,
        )
    except ValueError as exc:
        print(f"Qwen policy resolution failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        "label": resolution.label,
        "resolved_model_id": resolution.resolved_model_id,
        "tokenizer_id": resolution.tokenizer_id,
        "trust_remote_code": resolution.trust_remote_code,
        "dtype": resolution.dtype,
        "device": resolution.device,
        "is_resolved": resolution.is_resolved,
        "requires_manual_resolution": resolution.requires_manual_resolution,
        "notes": list(resolution.notes),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
