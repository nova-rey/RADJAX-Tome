"""C6 authority exports for the canonical production route.

The functions accept the facade's configuration structurally and never import
the facade, so selection and score stages can share this authority boundary
without a reverse dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from radjax_tome.builder.authority_hashes import (
    AUTHORITY_HASH_V2,
    AUTHORITY_MANIFEST_SCHEMA_V2,
    authority_hashes_for_artifact,
)
from radjax_tome.builder.c6_integration import (
    C6_SELECTION_INTEGRATION_POLICY,
    export_corridor_candidate_features,
    export_production_global_board_supply,
    export_production_source_passports,
)
from radjax_tome.builder.exemplar_selection import build_exemplar_selection_manifest
from radjax_tome.builder.production_stages.evidence import native_file_evidence
from radjax_tome.builder.production_stages.shared import (
    exemplar_capture_mode,
    file_sha256,
    native_c6_path_b_enabled,
    selection_integration_hash,
)
from radjax_tome.builder.teacher_textbook import load_text_examples
from radjax_tome.io.json import read_json_object, write_json
from radjax_tome.targets.store import TeacherTargetStore


def export_c6_selection_authorities(config: Any) -> dict[str, Any]:
    fingerprint = export_c6_fingerprint_selection_authority(config)
    return export_c6_global_authority(config, fingerprint)


def export_c6_fingerprint_selection_authority(config: Any) -> dict[str, Any]:
    if config.total_selected_exemplar_budget is None:
        raise ValueError("C6 total_selected_exemplar_budget is required")
    c6_root = config.output_dir / "c6"
    c6_root.mkdir(parents=True, exist_ok=True)
    store = TeacherTargetStore.open(config.output_dir)
    examples = load_text_examples(
        config.dataset_path, max_examples=store.metadata.num_examples
    )
    selector_manifest = build_exemplar_selection_manifest(
        store,
        examples=examples,
        batch_size=max(1, config.shard_size_examples),
        capture_mode=exemplar_capture_mode(config),
        fulfillment_policy=(
            "rerun_selected_capture"
            if config.exemplar_delivery_path == "two_pass_rerun_selected"
            else "select_from_existing_capture"
        ),
        board_capacity=max(
            config.total_selected_exemplar_budget,
            config.exemplar_leaderboard_capacity,
        ),
        budget_examples=None,
        budget_fraction=None,
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        canonical_score_fields_only=True,
        use_score_pass_fields=True,
        production_global_selector=True,
    )
    selector_path = c6_root / "production_global_selector.json"
    write_json(selector_path, selector_manifest)
    authority_hashes = authority_hashes_for_artifact(
        config.output_dir,
        selection_integration_config_hash=selection_integration_hash(config),
    )
    score_pass_authority_hash = authority_hashes.score_pass_authority_hash_v2
    feature_path = export_corridor_candidate_features(
        artifact_dir=config.output_dir, output_dir=c6_root / "corridor-features"
    )
    feature_manifest_path = c6_root / "corridor-features" / "manifest.json"
    feature_manifest = read_json_object(feature_manifest_path)
    feature_manifest["score_pass_authority_hash"] = score_pass_authority_hash
    feature_manifest["score_pass_authority_contract_version"] = AUTHORITY_HASH_V2
    write_json(feature_manifest_path, feature_manifest)
    return {
        "selector_path": selector_path,
        "feature_path": feature_path,
        "score_pass_authority_hash": score_pass_authority_hash,
        "score_pass_authority_hash_v1": authority_hashes.score_pass_authority_hash_v1,
        "score_pass_authority_contract_version": AUTHORITY_HASH_V2,
        "raw_artifact_digests": authority_hashes.raw_artifact_digests,
    }


def export_c6_global_authority(
    config: Any, fingerprint: Mapping[str, Any]
) -> dict[str, Any]:
    c6_root = config.output_dir / "c6"
    selector_path = Path(str(fingerprint["selector_path"]))
    feature_path = Path(str(fingerprint["feature_path"]))
    score_pass_authority_hash = str(fingerprint["score_pass_authority_hash"])
    score_pass_authority_hash_v1 = str(fingerprint["score_pass_authority_hash_v1"])
    raw_artifact_digests = dict(fingerprint["raw_artifact_digests"])
    selector_manifest = read_json_object(selector_path)
    global_supply = export_production_global_board_supply(
        selector_manifest,
        source_artifact_id=str(config.output_dir),
        source_artifact_hash=score_pass_authority_hash,
    )
    global_supply["source_provenance"]["score_pass_authority_hash"] = (
        score_pass_authority_hash
    )
    global_path = c6_root / "global-board-supply.json"
    write_json(global_path, global_supply)
    passports_path = export_production_source_passports(
        artifact_dir=config.output_dir,
        output_path=c6_root / "source-passports.json",
        score_pass_authority_hash=score_pass_authority_hash,
    )
    authority_manifest_path = c6_root / "authority_manifest.json"
    run_manifest = read_json_object(config.output_dir / "run_manifest.json")
    authority_manifest = {
        "schema_version": AUTHORITY_MANIFEST_SCHEMA_V2,
        "score_pass_authority_contract_version": AUTHORITY_HASH_V2,
        "score_pass_authority_hash": score_pass_authority_hash,
        "score_pass_authority_hash_v1": score_pass_authority_hash_v1,
        "raw_artifact_digests": raw_artifact_digests,
        "target_store_metadata_sha256": raw_artifact_digests["metadata.json"],
        "corpus_hash": run_manifest.get("corpus_hash"),
        "score_pass_config_hash": run_manifest.get("emission_config_hash"),
        "score_pass_resume_hash": run_manifest.get("resume_config_hash"),
        "selection_integration_config_hash": selection_integration_hash(config),
        "delivery_path": config.exemplar_delivery_path,
        "paths": {
            "selector": selector_path.relative_to(config.output_dir).as_posix(),
            "global_board_supply": global_path.relative_to(
                config.output_dir
            ).as_posix(),
            "source_passports": passports_path.relative_to(
                config.output_dir
            ).as_posix(),
            "corridor_features": feature_path.relative_to(config.output_dir).as_posix(),
        },
        "hashes": {
            "selector_sha256": raw_artifact_digests[
                "c6/production_global_selector.json"
            ],
            "global_board_supply_sha256": file_sha256(global_path),
            "source_passports_manifest_sha256": file_sha256(passports_path),
            "corridor_features_sha256": file_sha256(feature_path),
        },
        "external_authority_override_used": False,
        "production_grade": True,
    }
    write_json(authority_manifest_path, authority_manifest)
    mark_native_c6_score_pass_artifact(config, selector_path, authority_manifest_path)
    external_override_used = validate_external_c6_overrides(
        config, score_pass_authority_hash
    )
    selected_global_path = config.global_board_supply_path or global_path
    selected_passports_path = config.source_passports_path or passports_path
    authority_manifest["authority_paths_used"] = {
        "global_board_supply": str(selected_global_path),
        "source_passports": str(selected_passports_path),
    }
    authority_manifest["external_authority_override_used"] = external_override_used
    write_json(authority_manifest_path, authority_manifest)
    return {
        "feature_path": feature_path,
        "global_board_supply_path": selected_global_path,
        "source_passports_path": selected_passports_path,
        "authority_manifest_path": authority_manifest_path,
        "score_pass_authority_hash": score_pass_authority_hash,
        "external_authority_override_used": external_override_used,
    }


def mark_native_c6_score_pass_artifact(
    config: Any, selector_path: Path, authority_manifest_path: Path
) -> None:
    if not native_c6_path_b_enabled(config):
        return
    path = config.output_dir / "emission_config.json"
    payload = read_json_object(path)
    payload["claims_not_made"] = [
        str(item)
        for item in payload.get("claims_not_made", ())
        if str(item) != "no_production_global_two_pass_selector"
    ]
    payload.update(
        {
            "exemplar_selection_enabled": True,
            "exemplar_selection_manifest": selector_path.relative_to(
                config.output_dir
            ).as_posix(),
            "selection_integration_policy": C6_SELECTION_INTEGRATION_POLICY,
            "native_execution_mode": "native_c6_path_b_v1",
            "selection_authority_manifest": authority_manifest_path.relative_to(
                config.output_dir
            ).as_posix(),
        }
    )
    write_json(path, payload)


def validate_external_c6_overrides(config: Any, score_pass_authority_hash: str) -> bool:
    used = False
    for label, path in (
        ("global board supply", config.global_board_supply_path),
        ("source passports", config.source_passports_path),
    ):
        if path is None:
            continue
        if not path.is_file():
            raise ValueError(f"{label} override path missing: {path}")
        payload = read_json_object(path)
        provenance = payload.get("source_provenance", payload)
        observed = (
            provenance.get("score_pass_authority_hash")
            if isinstance(provenance, Mapping)
            else None
        )
        if observed != score_pass_authority_hash:
            raise ValueError(
                f"{label} override does not match the current score-pass authority hash"
            )
        used = True
    return used


def native_fingerprint_authority_operation(state: Any) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageResult

    state.progress.stage("selection_authority_export")
    fingerprint = export_c6_fingerprint_selection_authority(state.config)
    evidence = native_file_evidence(
        "fingerprint_corridor_selection_authority_export",
        (
            Path(str(fingerprint["selector_path"])),
            Path(str(fingerprint["feature_path"])),
            state.config.output_dir / "c6" / "corridor-features" / "manifest.json",
        ),
    )
    return StageResult(status="pass", value=fingerprint, evidence=evidence)


def native_global_authority_operation(
    state: Any, fingerprint: Mapping[str, Any]
) -> Any:
    from radjax_tome.builder.native_path_b.contracts import StageResult

    authorities = export_c6_global_authority(state.config, fingerprint)
    evidence = native_file_evidence(
        "global_authority_export",
        (
            Path(str(authorities["global_board_supply_path"])),
            Path(str(authorities["source_passports_path"])),
            Path(str(authorities["authority_manifest_path"])),
        ),
    )
    state.progress.memory_checkpoint("authority_export_complete")
    return StageResult(status="pass", value=authorities, evidence=evidence)
