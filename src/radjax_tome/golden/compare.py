from __future__ import annotations

import json
import tempfile
from itertools import zip_longest
from pathlib import Path
from typing import Any

from radjax_tome.golden.projection import capture_golden_contract, validate_fixture
from radjax_tome.quantization import ENTROPY_PARITY_QUANTIZATION_STEP


def compare_fixture_artifact(fixture_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    validate_fixture(fixture_dir)
    with tempfile.TemporaryDirectory(prefix="radjax-golden-observed-") as temporary:
        capture_golden_contract(artifact_dir, Path(temporary))
        return compare_contracts(fixture_dir, Path(temporary))


def compare_contracts(expected_dir: Path, observed_dir: Path) -> dict[str, Any]:
    from radjax_tome.golden.projection import _read_object

    expected = _read_object(expected_dir / "contract.json")
    observed = _read_object(observed_dir / "contract.json")
    if expected.get("schema_version") != observed.get("schema_version"):
        return {"status": "incompatible", "differences": ["schema_version"]}
    differences: list[dict[str, Any]] = []
    for field in ("input_identity", "semantic_policy", "board_summary_digest"):
        if expected.get(field) != observed.get(field):
            differences.append(
                {
                    "collection": "contract",
                    "field": field,
                    "expected": expected.get(field),
                    "observed": observed.get(field),
                }
            )
    for name in ("selected_obligations", "source_passports", "payload_semantics"):
        differences.extend(
            _compare_jsonl_rows(
                name,
                expected_dir / f"{name}.jsonl",
                observed_dir / f"{name}.jsonl",
            )
        )
    return {
        "status": "pass" if not differences else "fail",
        "expected_semantic_root": expected.get("semantic_root"),
        "observed_semantic_root": observed.get("semantic_root"),
        "differences": differences,
        "storage_only_differences": [],
    }


def _compare_jsonl_rows(
    name: str,
    expected_path: Path,
    observed_path: Path,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for row_number, (left, right) in enumerate(
        zip_longest(_iter_jsonl(expected_path), _iter_jsonl(observed_path)), start=1
    ):
        if left is None or right is None:
            differences.append(
                {
                    "collection": name,
                    "field": "count",
                    "row_number": row_number,
                    "expected": "record" if left is not None else None,
                    "observed": "record" if right is not None else None,
                }
            )
            continue
        coordinate = (left.get("selected_example_id"), left.get("selected_position"))
        if coordinate != (
            right.get("selected_example_id"),
            right.get("selected_position"),
        ):
            differences.append(
                {
                    "collection": name,
                    "coordinate": coordinate,
                    "field": "coordinate",
                    "observed": (
                        right.get("selected_example_id"),
                        right.get("selected_position"),
                    ),
                }
            )
            continue
        for key in sorted(set(left) | set(right)):
            if key == "teacher_entropy" and key in left and key in right:
                delta = abs(float(left[key]) - float(right[key]))
                if delta <= ENTROPY_PARITY_QUANTIZATION_STEP:
                    continue
                differences.append(
                    {
                        "collection": name,
                        "coordinate": coordinate,
                        "field": key,
                        "expected": left[key],
                        "observed": right[key],
                        "delta": delta,
                        "tolerance": ENTROPY_PARITY_QUANTIZATION_STEP,
                    }
                )
            elif left.get(key) != right.get(key):
                differences.append(
                    {
                        "collection": name,
                        "coordinate": coordinate,
                        "field": key,
                        "expected": left.get(key),
                        "observed": right.get(key),
                    }
                )
    return differences


def _iter_jsonl(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(
                        f"golden comparison expected object record: {path}"
                    )
                yield value
