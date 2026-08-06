"""Synthetic no-inference runner for the private provenance-shape experiment.

It expands the committed selected-record fixture deterministically.  It is not
a public CLI, a production writer, or a model-performance benchmark.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from experiments.provenance_shape_bakeoff import (
    SCHEMA,
    build_projection,
    validate_standard_projection,
)

AUTHORITY = {
    "teacher": "declared",
    "tokenizer": "declared",
    "selection": "fixed",
}
CONTRACT_VERSION = "experimental-contract-vnext"
BEHAVIORAL_POLICY = "experimental-behavior-policy-vnext"


def load_seed(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def deterministic_records(
    seed: list[dict[str, Any]], count: int
) -> Iterator[dict[str, Any]]:
    """Yield uniquely identified synthetic logical records without inference."""
    if not seed or count < 1:
        raise ValueError("benchmark requires a nonempty seed and positive count")
    for index in range(count):
        record = copy.deepcopy(seed[index % len(seed)])
        record["selected_example_id"] = f"synthetic_{index:08d}"
        record["selected_position"] = index
        record["source_position"] = index
        payload_ref = record.get("payload_ref")
        if isinstance(payload_ref, dict):
            payload_ref["source_position"] = index
        yield record


def run_once(
    *, seed_path: Path, count: int, capacity: int, shape: str
) -> dict[str, Any]:
    """Build and validate one disposable projection in a fresh temporary root."""
    seed_bytes = seed_path.read_bytes()
    seed = load_seed(seed_path)
    with tempfile.TemporaryDirectory(
        prefix="radjax-provenance-shape-benchmark-"
    ) as temp:
        result = build_projection(
            deterministic_records(seed, count),
            Path(temp) / shape,
            authority=AUTHORITY,
            capacity=capacity,
            shape=shape,
            contract_version=CONTRACT_VERSION,
            behavioral_policy_identity=BEHAVIORAL_POLICY,
        )
        validation_started = time.perf_counter()
        validated_count = len(
            list(validate_standard_projection(result.root, authority=AUTHORITY))
        )
        validation_seconds = time.perf_counter() - validation_started
    return {
        "schema_version": SCHEMA + ".synthetic_benchmark.v1",
        "shape": shape,
        "record_count": count,
        "shard_capacity": capacity,
        "seed_sha256": "sha256:" + hashlib.sha256(seed_bytes).hexdigest(),
        "semantic_root": result.semantic_root,
        "sequence_digest": result.sequence_digest,
        "archive_digest": result.archive_digest,
        "construction_seconds": result.construction_seconds,
        "archive_seconds": result.archive_seconds,
        "validation_seconds": validation_seconds,
        "throughput_records_per_second": count / result.construction_seconds,
        "validated_count": validated_count,
        "peak_rss_bytes": result.peak_rss_bytes,
        "counters": result.counters,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--shard-capacity", type=int, default=64)
    parser.add_argument("--shape", choices=("current", "candidate"), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            run_once(
                seed_path=args.seed,
                count=args.records,
                capacity=args.shard_capacity,
                shape=args.shape,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess.
    raise SystemExit(main())
