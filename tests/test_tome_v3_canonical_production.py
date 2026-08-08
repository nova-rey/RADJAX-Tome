from __future__ import annotations

from pathlib import Path

from radjax_contract.tome.v3.validation import validate_tome_artifact_v3

from radjax_tome.builder.production import build_production_gpu_tome
from tests.test_m4_live_canonical_execution import _canonical_config


def test_canonical_production_emits_opt_in_v3_from_finalized_handoff(
    tmp_path: Path,
) -> None:
    config = _canonical_config(
        tmp_path,
        artifact_contract_version="v3",
        payload_records_per_shard=2,
    )

    report = build_production_gpu_tome(config)

    assert report["status"] == "pass", report["blockers"]
    assert report["canonical_tome_contract_id"] == "radjax_tome_artifact_contract"
    assert report["canonical_tome_contract_version"] == "3.0.0"
    assert report["canonical_tome_selected_count"] == 4
    assert report["canonical_tome_shard_count"] == 2
    assert report["canonical_tome_transport_pair_atomicity"] is False

    directory = validate_tome_artifact_v3(Path(report["canonical_tome_directory"]))
    archive = validate_tome_artifact_v3(Path(report["canonical_tome_archive"]))
    assert directory.ok and archive.ok
    assert directory.semantic_root == archive.semantic_root
    assert directory.record_count == archive.record_count == 4
