"""Materialize explicit native-v3 Student-consumption v3 sidecar evidence.

V3 is an opt-in successor to the published v2 profile.  It reuses v2's
deterministic derived-resource construction, but binds the three evidence
sidecars whose semantics v2 did not close: row ranges, delivery receipt, and
authority reference.  Existing v2 artifacts are neither rewritten nor
reinterpreted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.io.json import read_json_object
from radjax_tome.tome.student_consumption_v2 import (
    ASSIGNMENT_PATH as _V2_ASSIGNMENT_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    AUTHORITY_REFERENCE_PATH as _V2_AUTHORITY_REFERENCE_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    DELIVERY_RECEIPT_PATH as _V2_DELIVERY_RECEIPT_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    EXAMPLE_REGISTRY_PATH as _V2_EXAMPLE_REGISTRY_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    OBSERVED_STATISTICS_PATH as _V2_OBSERVED_STATISTICS_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    ROW_RANGES_PATH as _V2_ROW_RANGES_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    SELECTED_EXEMPLAR_PAYLOAD_PATH as _V2_SELECTED_EXEMPLAR_PAYLOAD_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    SELECTED_PASSPORT_INDEX_PATH as _V2_SELECTED_PASSPORT_INDEX_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    TARGET_ROWS_PATH as _V2_TARGET_ROWS_PATH,
)
from radjax_tome.tome.student_consumption_v2 import (
    materialize_native_v3_student_consumption_v2,
)

SIDECAR_DIRECTORY = "student_consumption/v3"
ASSIGNMENT_PATH = _V2_ASSIGNMENT_PATH.replace("/v2/", "/v3/")
OBSERVED_STATISTICS_PATH = _V2_OBSERVED_STATISTICS_PATH.replace("/v2/", "/v3/")
TARGET_ROWS_PATH = _V2_TARGET_ROWS_PATH.replace("/v2/", "/v3/")
EXAMPLE_REGISTRY_PATH = _V2_EXAMPLE_REGISTRY_PATH.replace("/v2/", "/v3/")
ROW_RANGES_PATH = _V2_ROW_RANGES_PATH.replace("/v2/", "/v3/")
DELIVERY_RECEIPT_PATH = _V2_DELIVERY_RECEIPT_PATH.replace("/v2/", "/v3/")
AUTHORITY_REFERENCE_PATH = _V2_AUTHORITY_REFERENCE_PATH.replace("/v2/", "/v3/")
SELECTED_PASSPORT_INDEX_PATH = _V2_SELECTED_PASSPORT_INDEX_PATH.replace("/v2/", "/v3/")
SELECTED_EXEMPLAR_PAYLOAD_PATH = _V2_SELECTED_EXEMPLAR_PAYLOAD_PATH.replace(
    "/v2/", "/v3/"
)


@dataclass(frozen=True)
class NativeV3StudentConsumptionV3Materialization:
    """V3 paths and counts emitted into a package staging directory."""

    root: Path
    role_paths: dict[str, str]
    example_count: int
    assignment_count: int


def materialize_native_v3_student_consumption_v3(
    artifact_root: Path,
    *,
    destination_root: Path | None = None,
    selection_integration_config_hash: str | None = None,
) -> NativeV3StudentConsumptionV3Materialization:
    """Write the closed v3 evidence sidecar from native production facts."""

    source_root = Path(artifact_root)
    root = source_root if destination_root is None else Path(destination_root)
    v2 = materialize_native_v3_student_consumption_v2(
        source_root, destination_root=root
    )
    v2_dir = root / "student_consumption" / "v2"
    v3_dir = root / SIDECAR_DIRECTORY
    if v3_dir.exists():
        raise ValueError("native v3 Student-consumption v3 sidecar already exists")
    v2_dir.replace(v3_dir)

    _write_json(
        v3_dir / "row_ranges.json",
        {
            "schema_version": "native_v3_student_consumption_row_ranges_v1",
            "example_count": v2.example_count,
            "assignment_count": v2.assignment_count,
            "ordering": "example_index_then_source_position",
        },
    )
    _write_json(v3_dir / "delivery_receipt.json", _delivery_receipt(source_root))
    _write_json(
        v3_dir / "authority_reference.json",
        _authority_reference(
            source_root,
            selection_integration_config_hash=selection_integration_config_hash,
        ),
    )
    return NativeV3StudentConsumptionV3Materialization(
        root=root,
        role_paths={
            key: value.replace("/v2/", "/v3/") for key, value in v2.role_paths.items()
        },
        example_count=v2.example_count,
        assignment_count=v2.assignment_count,
    )


def _delivery_receipt(source_root: Path) -> dict[str, Any]:
    delivery = _first_object(
        source_root,
        "delivery_report.json",
        "production_build_report.json",
        "c6/authority_manifest.json",
    )
    delivery_path = {
        "one_pass_pruned_candidate": "one_pass_full",
        "one_pass_full": "one_pass_full",
        "two_pass_rerun_selected": "two_pass_rerun_selected",
    }.get(delivery.get("delivery_path"))
    if delivery_path not in {"one_pass_full", "two_pass_rerun_selected"}:
        raise ValueError("native v3 artifact has no recognized delivery_path evidence")
    return {
        "schema_version": "native_v3_student_consumption_delivery_receipt_v2",
        "delivery_path": delivery_path,
        "assignment_encoding": "npz_named_arrays_v1",
        "statistics_encoding": "npz_named_arrays_v1",
        "source_roles": ["native_v3_mode_assignments", "native_v3_score_shards"],
    }


def _authority_reference(
    source_root: Path, *, selection_integration_config_hash: str | None
) -> dict[str, Any]:
    authority: dict[str, Any] = {}
    for relative in (
        "c6/authority_manifest.json",
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
                if document.get(key) is not None and key not in authority:
                    authority[key] = document[key]
    if selection_integration_config_hash is not None:
        existing = authority.get("selection_integration_config_hash")
        if existing not in {None, selection_integration_config_hash}:
            raise ValueError("native v3 selection authority disagrees with identity")
        authority["selection_integration_config_hash"] = (
            selection_integration_config_hash
        )
    if "selection_integration_config_hash" not in authority:
        raise ValueError("native v3 artifact has no selection integration authority")
    if (
        not {"score_pass_authority_hash", "score_pass_authority_hash_v1"}
        & authority.keys()
    ):
        raise ValueError("native v3 artifact has no score-pass authority")
    return {
        "schema_version": "native_v3_student_consumption_authority_reference_v1",
        **authority,
    }


def _first_object(root: Path, *relatives: str) -> dict[str, Any]:
    for relative in relatives:
        candidate = root / relative
        if candidate.is_file():
            return read_json_object(candidate)
    raise ValueError("native v3 artifact has no delivery evidence")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
