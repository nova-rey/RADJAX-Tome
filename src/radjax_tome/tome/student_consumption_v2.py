"""Materialize native-v3 Student-consumption v2 sidecar inputs.

This adapter deliberately does *not* edit a native-v3 cover, manifest, or
identity.  It converts already-produced native-v3 source facts into a small,
explicit sidecar that the packaging boundary can inventory and bind to the
Contract-owned ``native_v3_student_v2`` profile.  Keeping this operation
separate is what lets the historic v3 semantic root remain historical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radjax_tome.io.json import read_json_object

SIDECAR_DIRECTORY = "student_consumption/v2"
ASSIGNMENT_PATH = f"{SIDECAR_DIRECTORY}/corridor_assignments.npz"
OBSERVED_STATISTICS_PATH = f"{SIDECAR_DIRECTORY}/corridor_observed_statistics.npz"
TARGET_ROWS_PATH = f"{SIDECAR_DIRECTORY}/target_rows.npz"
EXAMPLE_REGISTRY_PATH = f"{SIDECAR_DIRECTORY}/example_registry.json"
ROW_RANGES_PATH = f"{SIDECAR_DIRECTORY}/row_ranges.json"
DELIVERY_RECEIPT_PATH = f"{SIDECAR_DIRECTORY}/delivery_receipt.json"
AUTHORITY_REFERENCE_PATH = f"{SIDECAR_DIRECTORY}/authority_reference.json"
SELECTED_PASSPORT_INDEX_PATH = f"{SIDECAR_DIRECTORY}/selected_passport_index.json"
SELECTED_EXEMPLAR_PAYLOAD_PATH = f"{SIDECAR_DIRECTORY}/selected_exemplar_payload.json"

_ASSIGNMENT_ARRAYS = ("position_example_index", "position", "mode_id", "weight")
_STAT_ARRAYS = (
    "corridor_entropy",
    "corridor_top1_margin",
    "corridor_top8_mass",
    "corridor_top32_mass",
    "corridor_tail_mass",
)


@dataclass(frozen=True)
class NativeV3StudentConsumptionV2Materialization:
    """Paths and role declarations written by :func:`materialize...`.

    ``role_paths`` are artifact-relative physical locators only.  They are not
    a semantic identity recipe; Contract v0.4.1 owns that recipe and binds
    these files through the package inventory.
    """

    root: Path
    role_paths: dict[str, str]
    example_count: int
    assignment_count: int


def materialize_native_v3_student_consumption_v2(
    artifact_root: Path,
    *,
    destination_root: Path | None = None,
) -> NativeV3StudentConsumptionV2Materialization:
    """Write deterministic v2 sidecar resources from a native-v3 artifact.

    The artifact must retain its native source shards.  The package stage calls
    this before profile reduction, while the full source facts are still
    available.  All assignments are emitted as NPZ arrays; legacy JSON
    assignment documents remain source evidence and are never offered as a v2
    assignment resource.
    """

    source_root = Path(artifact_root)
    root = source_root if destination_root is None else Path(destination_root)
    assignment_document = read_json_object(root / "corridors" / "mode_assignments.json")
    arrays = assignment_document.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("native v3 mode assignments have no array declarations")
    assignment_arrays = {
        name: _load_declared_array(root, arrays, name) for name in _ASSIGNMENT_ARRAYS
    }
    _validate_assignment_arrays(assignment_arrays)
    example_ids = _read_example_ids(root, assignment_document)
    if len(example_ids) != int(assignment_document.get("num_examples") or -1):
        raise ValueError("native v3 examples metadata count disagrees with assignments")

    source_rows = _source_rows(source_root, expected_example_ids=example_ids)
    input_ids = np.stack([row["input_ids"] for row in source_rows], axis=0)
    attention_mask = np.stack([row["attention_mask"] for row in source_rows], axis=0)
    lengths = np.asarray(
        [int(row["corridor_lengths"]) for row in source_rows], dtype=np.int32
    )
    statistics = _assignment_statistics(
        assignment_arrays=assignment_arrays,
        source_rows=source_rows,
    )

    sidecar = root / SIDECAR_DIRECTORY
    sidecar.mkdir(parents=True, exist_ok=True)
    np.savez(
        sidecar / "corridor_assignments.npz",
        **{
            "position_example_index": assignment_arrays[
                "position_example_index"
            ].astype(np.int32, copy=False),
            "position": assignment_arrays["position"].astype(np.int32, copy=False),
            "mode_id": assignment_arrays["mode_id"].astype(np.int32, copy=False),
            "weight": assignment_arrays["weight"].astype(np.float32, copy=False),
        },
    )
    np.savez(sidecar / "corridor_observed_statistics.npz", **statistics)
    np.savez(
        sidecar / "target_rows.npz",
        input_ids=input_ids.astype(_integer_dtype(input_ids), copy=False),
        attention_mask=attention_mask.astype(np.int32, copy=False),
        corridor_lengths=lengths,
    )
    _write_example_registry(sidecar / "example_registry.json", example_ids)
    _write_json(
        sidecar / "row_ranges.json",
        {
            "schema_version": "native_v3_student_consumption_row_ranges_v1",
            "example_count": len(example_ids),
            "assignment_count": int(assignment_arrays["position"].shape[0]),
            "ordering": "example_index_then_source_position",
        },
    )
    _write_json(
        sidecar / "delivery_receipt.json",
        {
            "schema_version": "native_v3_student_consumption_delivery_receipt_v1",
            "assignment_source": "corridors/mode_assignments.json",
            "source_shards": "shards/shard-*.npz",
            "assignment_encoding": "npz_named_arrays_v1",
            "statistics_encoding": "npz_named_arrays_v1",
        },
    )
    _write_authority_reference(source_root, sidecar / "authority_reference.json")
    _materialize_selected_resources(root, sidecar)
    return NativeV3StudentConsumptionV2Materialization(
        root=root,
        role_paths={
            "target_shard": TARGET_ROWS_PATH,
            "example_registry": EXAMPLE_REGISTRY_PATH,
            "corridor_mode_table": "corridors/corridor_modes.json",
            "corridor_assignment": ASSIGNMENT_PATH,
            "selected_passport_index": SELECTED_PASSPORT_INDEX_PATH,
            "selected_exemplar_payload": SELECTED_EXEMPLAR_PAYLOAD_PATH,
            "corridor_observed_statistics": OBSERVED_STATISTICS_PATH,
            "row_range_declaration": ROW_RANGES_PATH,
            "delivery_receipt": DELIVERY_RECEIPT_PATH,
            "authority_reference": AUTHORITY_REFERENCE_PATH,
        },
        example_count=len(example_ids),
        assignment_count=int(assignment_arrays["position"].shape[0]),
    )


def _load_declared_array(root: Path, arrays: dict[str, Any], name: str) -> np.ndarray:
    spec = arrays.get(name)
    if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
        raise ValueError(f"native v3 mode assignments missing array {name}")
    path = root / str(spec["path"])
    if not path.is_file():
        raise ValueError(f"native v3 mode assignment array is missing: {name}")
    return np.asarray(np.load(path, allow_pickle=False))


def _validate_assignment_arrays(arrays: dict[str, np.ndarray]) -> None:
    count = arrays["position"].shape[0]
    if count == 0 or any(
        array.ndim != 1 or array.shape[0] != count for array in arrays.values()
    ):
        raise ValueError("native v3 assignment arrays must be nonempty aligned vectors")
    if any(
        not np.issubdtype(arrays[name].dtype, np.integer)
        for name in _ASSIGNMENT_ARRAYS[:3]
    ):
        raise ValueError(
            "native v3 assignment coordinates and mode IDs must be integers"
        )
    if not np.issubdtype(arrays["weight"].dtype, np.floating):
        raise ValueError("native v3 assignment weights must be floating point")
    if (arrays["position_example_index"] < 0).any() or (arrays["position"] < 0).any():
        raise ValueError("native v3 assignment coordinates must be nonnegative")
    if not np.isfinite(arrays["weight"]).all() or (arrays["weight"] < 0).any():
        raise ValueError("native v3 assignment weights must be finite and nonnegative")


def _read_example_ids(root: Path, assignment: dict[str, Any]) -> list[str]:
    metadata = assignment.get("examples_metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise ValueError("native v3 mode assignments have no examples metadata")
    path = root / str(metadata["path"])
    result: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for expected_index, line in enumerate(lines):
        entry = json.loads(line)
        if not isinstance(entry, dict) or entry.get("example_index") != expected_index:
            raise ValueError(
                "native v3 examples metadata is not deterministically ordered"
            )
        example_id = entry.get("example_id")
        if not isinstance(example_id, str) or not example_id or example_id in result:
            raise ValueError(
                "native v3 examples metadata has invalid example identities"
            )
        result.append(example_id)
    return result


def _source_rows(
    root: Path, *, expected_example_ids: list[str]
) -> list[dict[str, np.ndarray]]:
    keyed: dict[str, dict[str, np.ndarray]] = {}
    ordered: list[dict[str, np.ndarray]] = []
    for path in sorted((root / "shards").glob("shard-*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            required = (
                "input_ids",
                "attention_mask",
                "score_example_ids",
                "corridor_lengths",
                *_STAT_ARRAYS,
            )
            missing = [name for name in required if name not in shard.files]
            if missing:
                raise ValueError(
                    "source shard lacks v2 sidecar evidence: " + ", ".join(missing)
                )
            ids = np.asarray(shard["score_example_ids"])
            for row in range(np.asarray(shard["input_ids"]).shape[0]):
                example_id = _decode_example_id(ids[row])
                item = {
                    "input_ids": np.asarray(shard["input_ids"])[row],
                    "attention_mask": np.asarray(shard["attention_mask"])[row],
                    "corridor_lengths": np.asarray(shard["corridor_lengths"])[row],
                    **{name: np.asarray(shard[name])[row] for name in _STAT_ARRAYS},
                }
                # Some historical producer shards use shard-local numeric
                # score IDs. They cannot join the package registry globally;
                # retain source order for the existing cardinality-checked
                # fallback instead of treating that legacy local ID as a
                # portable identity.
                if example_id not in keyed:
                    keyed[example_id] = item
                ordered.append(item)
    if all(example_id in keyed for example_id in expected_example_ids):
        return [keyed[example_id] for example_id in expected_example_ids]
    if len(ordered) == len(expected_example_ids):
        return ordered
    raise ValueError("source shards cannot resolve every native-v3 example identity")


def _assignment_statistics(
    *,
    assignment_arrays: dict[str, np.ndarray],
    source_rows: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    example_indexes = assignment_arrays["position_example_index"]
    positions = assignment_arrays["position"]
    if int(example_indexes.max(initial=-1)) >= len(source_rows):
        raise ValueError("native v3 assignments reference an unknown example row")
    values: dict[str, np.ndarray] = {}
    for name in _STAT_ARRAYS:
        rows: list[float] = []
        for example_index, position in zip(example_indexes, positions, strict=True):
            stat = source_rows[int(example_index)][name]
            if int(position) >= stat.shape[0]:
                raise ValueError(f"native v3 assignment position is outside {name}")
            rows.append(float(stat[int(position)]))
        values[name.removeprefix("corridor_")] = np.asarray(rows, dtype=np.float32)
    return values


def _write_example_registry(path: Path, example_ids: list[str]) -> None:
    _write_json(
        path,
        {
            "examples": [
                {"global_example_index": index, "selected_example_id": example_id}
                for index, example_id in enumerate(example_ids)
            ]
        },
    )


def _write_authority_reference(source_root: Path, path: Path) -> None:
    """Copy known production authority facts without manufacturing authority."""
    authority: dict[str, Any] = {}
    for relative in (
        "production_build_report.json",
        "delivery_report.json",
        "metadata.json",
    ):
        candidate = source_root / relative
        if candidate.is_file():
            document = read_json_object(candidate)
            for key in (
                "score_pass_authority_hash",
                "score_pass_authority_hash_v1",
                "selection_integration_config_hash",
                "delivery_authority_hash",
            ):
                if key in document and key not in authority:
                    authority[key] = document[key]
    if not authority:
        raise ValueError("native v3 artifact has no production authority evidence")
    _write_json(path, authority)


def _materialize_selected_resources(root: Path, sidecar: Path) -> None:
    """Flatten native selected evidence into two raw-integrity-bindable files."""
    passport_path = root / "leaderboards" / "selected_exemplars.json"
    if not passport_path.is_file():
        raise ValueError("native v3 artifact has no selected passport index")
    passports = read_json_object(passport_path).get("selected_exemplars")
    if not isinstance(passports, list) or not all(
        isinstance(item, dict) for item in passports
    ):
        raise ValueError("native v3 selected passport index is invalid")
    exemplar_rows: list[dict[str, Any]] = []
    payload_dir = root / "selected_exemplars"
    for path in sorted(payload_dir.glob("selected-exemplars-*.json")):
        rows = read_json_object(path).get("selected_exemplars")
        if not isinstance(rows, list) or not all(
            isinstance(item, dict) for item in rows
        ):
            raise ValueError(f"native v3 selected payload is invalid: {path.name}")
        exemplar_rows.extend(rows)
    # The profile retains both exemplar roles even when a legitimate producer
    # selection contains no rows.  An empty pair is meaningful; only an
    # asymmetric passport/payload pair is an invalid native delivery.
    if bool(passports) != bool(exemplar_rows):
        raise ValueError("native v3 selected passport/payload presence disagrees")
    indexed = list(enumerate(exemplar_rows))
    indexed.sort(
        key=lambda item: (
            int(item[1].get("selection_index"))
            if isinstance(item[1].get("selection_index"), int)
            else item[0],
            item[0],
        )
    )
    passports_by_coordinate = {
        _selected_coordinate(record): record for record in passports
    }
    if None in passports_by_coordinate or len(passports_by_coordinate) != len(
        passports
    ):
        raise ValueError("native v3 selected passport has no stable coordinate")
    normalized_passports: list[dict[str, Any]] = []
    normalized_exemplars: list[dict[str, Any]] = []
    for rank, (_, record) in enumerate(indexed, start=1):
        coordinate = _selected_coordinate(record)
        passport = passports_by_coordinate.get(coordinate)
        if coordinate is None or passport is None:
            raise ValueError("native v3 selected payload cannot join passport")
        normalized_passport = dict(passport)
        normalized_exemplar = dict(record)
        normalized_passport["rank"] = rank
        normalized_exemplar["rank"] = rank
        normalized_passports.append(normalized_passport)
        normalized_exemplars.append(normalized_exemplar)
    _write_json(
        sidecar / "selected_passport_index.json",
        {"selected_exemplars": normalized_passports},
    )
    _write_json(
        sidecar / "selected_exemplar_payload.json",
        {"selected_exemplars": normalized_exemplars},
    )


def _selected_coordinate(record: dict[str, Any]) -> tuple[str, int] | None:
    example_id = record.get("selected_example_id")
    position = record.get("selected_position")
    if (
        not isinstance(example_id, str)
        or not example_id
        or isinstance(position, bool)
        or not isinstance(position, int)
        or position < 0
    ):
        return None
    return example_id, position


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _integer_dtype(values: np.ndarray) -> np.dtype[Any]:
    if (
        np.issubdtype(values.dtype, np.integer)
        and values.max(initial=0) <= np.iinfo(np.int32).max
    ):
        return np.dtype(np.int32)
    raise ValueError("native v3 token IDs cannot be represented as Contract int32")


def _decode_example_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
