from __future__ import annotations

import json
from pathlib import Path

from radjax_tome.tome.canonical_artifact import (
    build_canonical_artifact_cover,
    derive_tome_semantic_identity,
    validate_canonical_artifact_directory,
)


def _write(root: Path, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _artifact(root: Path, *, created_at: str) -> None:
    _write(
        root,
        "metadata.json",
        {
            "created_at": created_at,
            "target_type": "corridor_exemplar_v1",
            "sequence_length": 128,
            "vocab_size": 262144,
            "tome_version": 1,
        },
    )
    for relative in (
        "vocab_contract.json",
        "teacher_manifest.json",
        "emission_config.json",
        "corridors/corridor_summary.json",
        "corridors/corridor_modes.json",
        "corridors/mode_assignments.json",
        "leaderboards/selected_exemplars.json",
    ):
        _write(root, relative, {"schema_version": "test", "created_at": created_at})
    _write(root, "validation_report.json", {"status": "pass"})


def test_m5d_identity_ignores_runtime_created_at_but_raw_integrity_does_not(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _artifact(first, created_at="2026-07-19T00:00:00Z")
    _artifact(second, created_at="2026-07-24T00:00:00Z")

    first_identity = derive_tome_semantic_identity(first)
    second_identity = derive_tome_semantic_identity(second)
    first_cover = build_canonical_artifact_cover(
        first, profile="unpacked", transport="directory"
    )
    second_cover = build_canonical_artifact_cover(
        second, profile="unpacked", transport="directory"
    )

    assert first_identity.semantic_digest == second_identity.semantic_digest
    assert (
        first_cover["provenance"]["raw_artifact_digests"]["metadata.json"]
        != second_cover["provenance"]["raw_artifact_digests"]["metadata.json"]
    )
    validate_canonical_artifact_directory(first, first_cover)
    validate_canonical_artifact_directory(second, second_cover)
