#!/usr/bin/env python3
"""Validate a durable M8G workload manifest without resolving private paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "m8g_benchmark_workload_bundle_v1"
REQUIRED = {
    "schema_version",
    "provenance",
    "normalized_inputs",
    "selected_sources",
    "selected_coordinates",
    "authorities",
    "expected_counts",
    "replay_policy",
}


def validate(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if set(document) != REQUIRED:
        raise ValueError("bundle fields are not closed")
    if document["schema_version"] != SCHEMA:
        raise ValueError("unsupported workload bundle schema")
    provenance = document["provenance"]
    if not isinstance(provenance, dict) or provenance.get("status") not in {
        "HISTORICAL_M8_WORKLOAD_RECOVERED",
        "NEW_PAIRED_M8G_WORKLOAD_ESTABLISHED",
        "BLOCKED_NO_REPRODUCIBLE_WORKLOAD",
    }:
        raise ValueError("invalid workload provenance status")
    sources = document["selected_sources"]
    coordinates = document["selected_coordinates"]
    if not isinstance(sources, dict) or not isinstance(coordinates, dict):
        raise ValueError("selected identity manifests must be objects")
    source_count = int(sources["count"])
    coordinate_count = int(coordinates["count"])
    expected = document["expected_counts"]
    if not isinstance(expected, dict):
        raise ValueError("expected_counts must be an object")
    if expected.get("selected_coordinates") != coordinate_count:
        raise ValueError("selected coordinate count mismatch")
    if expected.get("selected_sources") != source_count:
        raise ValueError("selected source count mismatch")
    policy = document["replay_policy"]
    if not isinstance(policy, dict) or policy != {
        "requested_batch_size": 8,
        "batching": "fixed_source_count",
        "oom_fallback": "halving_only",
        "length_bucketing": False,
    }:
        raise ValueError("replay policy is not the frozen M8 policy")
    # The manifest is portable; byte verification is enabled by passing the
    # separately governed artifact root to the CLI.
    artifact_root = getattr(validate, "artifact_root", None)
    if artifact_root is not None:
        layout = provenance.get("artifact_layout")
        if not isinstance(layout, dict):
            raise ValueError("artifact layout is missing")
        replay_root = artifact_root / str(layout["replay_root"])
        required = {
            "c6/claims/selected_coordinates.jsonl": "sha256:d29e9f5642a808e5a439ec21beadbfc3793c2c2c6f14467fe3c1a8447784c5cb",  # noqa: E501
            "c6/multi-role-selection/selected_exemplars.jsonl": "sha256:255d5507668f8c6eed6f220563cc8bd3641fa134d721f3d9d9047161d899b35c",  # noqa: E501
            "c6/source-passports.jsonl": "sha256:b23bc3d9dbfb9845a77efab3819e99c7d410b8243153d1ba81120adf3987226b",  # noqa: E501
            "c6/authority_manifest.json": "sha256:765f3f997c1c56ff4333f4ed8dd61ec9696dcd6bef27c149c6569858c828027f",  # noqa: E501
        }
        for relative, expected_digest in required.items():
            candidate = replay_root / relative
            if not candidate.is_file() or _sha256(candidate) != expected_digest:
                raise ValueError(f"artifact member digest mismatch: {relative}")
        coordinates = replay_root / "c6/claims/selected_coordinates.jsonl"
        selected = replay_root / "c6/multi-role-selection/selected_exemplars.jsonl"
        passports = replay_root / "c6/source-passports.jsonl"
        coordinate_rows = [
            json.loads(line) for line in coordinates.read_text().splitlines()
        ]
        selected_rows = [json.loads(line) for line in selected.read_text().splitlines()]
        passport_keys = {
            (str(row["example_id"]), int(row["position"]))
            for row in (json.loads(line) for line in passports.read_text().splitlines())
        }
        coordinate_keys = {
            (
                str(row.get("selected_example_id", row.get("example_id"))),
                int(row.get("selected_position", row.get("position"))),
            )
            for row in coordinate_rows
        }
        selected_keys = {
            (
                str(row.get("selected_example_id", row.get("example_id"))),
                int(row.get("selected_position", row.get("position"))),
            )
            for row in selected_rows
        }
        if len(coordinate_keys) != coordinate_count or selected_keys != coordinate_keys:
            raise ValueError("selected coordinate identity mismatch")
        if not coordinate_keys <= passport_keys:
            raise ValueError("selected coordinate lacks source passport")
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": SCHEMA,
        "status": provenance["status"],
        "selected_sources": source_count,
        "selected_coordinates": coordinate_count,
        "manifest_sha256": "sha256:" + hashlib.sha256(canonical).hexdigest(),
        "artifact_verified": artifact_root is not None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    validate.artifact_root = args.bundle_root.resolve() if args.bundle_root else None
    print(json.dumps(validate(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
