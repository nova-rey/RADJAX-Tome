from __future__ import annotations

import json
from pathlib import Path

import pytest

from radjax_tome.golden.contract import (
    build_contract,
    digest_active_payload_storage,
    semantic_digest,
)
from radjax_tome.golden.projection import (
    _input_identity,
    _is_local_storage_locator,
    _require_truth_gate_fields,
    _semantic_authority,
    _semantic_board_summary,
    _semantic_policy,
    _write_fixture,
    validate_fixture,
)


def test_input_identity_reads_prefixed_corpus_provenance_and_model_name(
    tmp_path: Path,
) -> None:
    _write_projection_sidecars(
        tmp_path,
        teacher_manifest={
            "teacher_model_provenance": {
                "model_name": "google/gemma-3-270m",
                "model_revision": "revision-1",
                "config_hash": "sha256:config",
                "tokenizer_hash": "sha256:tokenizer",
                "weights_hash": "sha256:weights",
                "model_directory_hash": "sha256:directory",
            },
            "corpus_provenance": {
                "source_corpus_hash": "sha256:corpus",
                "source_corpus_manifest_hash": "sha256:manifest",
                "source_corpus_normalization_policy": "normalize_v1",
                "source_corpus_chunking_policy": "chunk_v1",
                "source_corpus_deduplication_policy": "dedupe_v1",
            },
        },
    )

    identity = _input_identity(tmp_path)

    assert identity["teacher_identity"]["model_name"] == "google/gemma-3-270m"
    assert identity["corpus_hash"] == "sha256:corpus"
    assert identity["corpus_manifest_hash"] == "sha256:manifest"
    assert identity["normalization_policy"] == "normalize_v1"
    assert identity["chunking_policy"] == "chunk_v1"
    assert identity["deduplication_policy"] == "dedupe_v1"


def test_semantic_policy_resolves_native_delivery_and_rerun_aliases(
    tmp_path: Path,
) -> None:
    _write_projection_sidecars(
        tmp_path,
        emission={
            "teacher_backend": "gpu_torch",
            "runtime_mode": "cpu_gpu",
            "target_policy": "corridor_exemplar_v1",
            "native_execution_mode": "native_c6_path_b_v1",
            "selection_integration_policy": "c6_multi_role_v1",
            "exemplar_delivery_path": "one_pass_pruned_candidate",
            "dynamic_top_k_min": 1,
            "dynamic_top_k_max": 128,
            "dynamic_mass_threshold": 0.95,
            "num_buckets": 4,
            "retain_unselected_exemplar_payloads": False,
        },
    )
    reports = {
        "delivery_report.json": {
            "delivery_path": "two_pass_rerun_selected",
            "selected_rerun_requested_batch_size": 16,
        },
        "production_build_report.json": {"selected_rerun_batch_size": 8},
    }

    policy = _semantic_policy(tmp_path, reports, {})

    assert policy["delivery_path"] == "two_pass_rerun_selected"
    assert policy["selected_rerun_batch_size"] == 16

    reports["delivery_report.json"].pop("delivery_path")
    reports["delivery_report.json"].pop("selected_rerun_requested_batch_size")
    fallback = _semantic_policy(tmp_path, reports, {})
    assert fallback["delivery_path"] == "one_pass_pruned_candidate"
    assert fallback["selected_rerun_batch_size"] == 8


def test_truth_gate_rejects_required_null_fields() -> None:
    input_identity = {
        "teacher_identity": {"model_name": "google/gemma-3-270m"},
        "teacher_model_hashes": {
            "config_hash": "sha256:config",
            "tokenizer_hash": "sha256:tokenizer",
            "weights_hash": "sha256:weights",
            "model_directory_hash": "sha256:directory",
        },
        "corpus_hash": "sha256:corpus",
        "corpus_manifest_hash": "sha256:manifest",
        "normalization_policy": "normalize_v1",
        "chunking_policy": "chunk_v1",
        "deduplication_policy": "dedupe_v1",
    }
    policy = {
        "teacher_backend": "gpu_torch",
        "runtime_mode": "cpu_gpu",
        "target_policy": "corridor_exemplar_v1",
        "native_execution_mode": "native_c6_path_b_v1",
        "delivery_path": "two_pass_rerun_selected",
        "selection_integration_policy": "c6_multi_role_v1",
        "dynamic_top_k_min": 1,
        "dynamic_top_k_max": 128,
        "dynamic_mass_threshold": 0.95,
    }

    _require_truth_gate_fields(input_identity, policy)
    input_identity["corpus_manifest_hash"] = None
    input_identity["teacher_model_hashes"]["weights_hash"] = None
    policy["dynamic_top_k_max"] = None

    with pytest.raises(ValueError) as exc_info:
        _require_truth_gate_fields(input_identity, policy)
    message = str(exc_info.value)
    assert "input_identity.corpus_manifest_hash" in message
    assert "input_identity.teacher_model_hashes.weights_hash" in message
    assert "semantic_policy.dynamic_top_k_max" in message


def test_c4_storage_hash_changes_do_not_change_semantic_projection() -> None:
    semantic_records = {
        "selected_coordinates.jsonl": [
            {"example_id": "corpus_0001", "position": 7, "selection_index": 1}
        ]
    }
    first = _semantic_board_summary(
        {
            "authority": _semantic_authority(
                {
                    "schema_version": "c6_authority_manifest_v1",
                    "score_pass_authority_hash": "sha256:authority",
                    "paths": {"claim_manifest": "c6/claims/claim_manifest.json"},
                    "hashes": {"claim_manifest_sha256": "sha256:first"},
                }
            ),
            "c4_manifest": {
                "files": {"selected_coordinates.jsonl": {"sha256": "sha256:first"}}
            },
            "c4_semantic_records": semantic_records,
        }
    )
    second = _semantic_board_summary(
        {
            "authority": _semantic_authority(
                {
                    "schema_version": "c6_authority_manifest_v1",
                    "score_pass_authority_hash": "sha256:authority",
                    "paths": {"claim_manifest": "c6/claims/claim_manifest.json"},
                    "hashes": {"claim_manifest_sha256": "sha256:second"},
                }
            ),
            "c4_manifest": {
                "files": {"selected_coordinates.jsonl": {"sha256": "sha256:second"}}
            },
            "c4_semantic_records": semantic_records,
        }
    )

    assert first == second


def test_board_summary_removes_nested_storage_artifact_ids_but_keeps_logical_ids() -> (
    None
):
    left = _semantic_board_summary(_nested_artifact_provenance("/teamspace/left"))
    right = _semantic_board_summary(_nested_artifact_provenance("/teamspace/right"))

    assert semantic_digest("board-summary", left) == semantic_digest(
        "board-summary", right
    )
    assert (
        _contract_for_board_summary(left)["semantic_root"]
        == _contract_for_board_summary(right)["semantic_root"]
    )
    assert left["feature_provenance"] == {"logical_artifact_id": "c2-logical-v1"}
    assert left["source_leaderboard_provenance"] == {
        "logical_artifact_id": "leaderboard-logical-v1"
    }
    assert left["source_provenance"]["c4"]["claims"] == {
        "logical_artifact_id": "c4-claims-v1"
    }
    assert left["feature_derivation"] == {"logical_artifact_id": "c2-derive-v1"}


@pytest.mark.parametrize(
    "value",
    (
        "/teamspace/radjax/artifact",
        r"C:\rental\artifact",
        r"\\server\share\artifact",
        "file:///teamspace/radjax/artifact",
        "~/radjax/artifact",
    ),
)
def test_portability_gate_recognizes_local_storage_locators(value: str) -> None:
    assert _is_local_storage_locator(value)


def test_capture_writer_and_fixture_validation_reject_local_storage_paths(
    tmp_path: Path,
) -> None:
    obligations, passports, payloads = _fixture_rows()
    contract = build_contract(
        fixture_metadata={},
        input_identity={},
        semantic_policy={},
        stage_summary=[],
        selected_obligations=obligations,
        source_passports=passports,
        payload_semantics=payloads,
        board_summary={},
    )
    passports[0]["unrecognized_storage_locator"] = "/teamspace/radjax/source"

    with pytest.raises(ValueError, match="portability violation"):
        _write_fixture(
            tmp_path / "rejected", contract, obligations, passports, payloads, {}
        )

    passports[0].pop("unrecognized_storage_locator")
    fixture = tmp_path / "fixture"
    _write_fixture(fixture, contract, obligations, passports, payloads, {})
    passport_path = fixture / "source_passports.jsonl"
    row = json.loads(passport_path.read_text(encoding="utf-8"))
    row["unknown_locator"] = "file:///teamspace/radjax/source"
    passport_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="portability violation"):
        validate_fixture(fixture)


def _write_projection_sidecars(
    root: Path,
    *,
    teacher_manifest: dict[str, object] | None = None,
    emission: dict[str, object] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {"vocab_size": 262144, "sequence_length": 128, "num_examples": 1000}
        ),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "teacher_model_hashes": {
                    "config_hash": "sha256:config",
                    "tokenizer_hash": "sha256:tokenizer",
                    "weights_hash": "sha256:weights",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest or {}), encoding="utf-8"
    )
    (root / "emission_config.json").write_text(
        json.dumps(emission or {}), encoding="utf-8"
    )


def _nested_artifact_provenance(storage_root: str) -> dict[str, object]:
    return {
        "feature_provenance": {
            "source_artifact_id": f"{storage_root}/c2/features.jsonl",
            "logical_artifact_id": "c2-logical-v1",
        },
        "source_leaderboard_provenance": {
            "leaderboard_artifact_id": f"{storage_root}/c2/leaderboards.jsonl",
            "logical_artifact_id": "leaderboard-logical-v1",
        },
        "source_provenance": {
            "c4": {
                "claims": {
                    "source_artifact_id": f"{storage_root}/c4/claims.jsonl",
                    "logical_artifact_id": "c4-claims-v1",
                }
            }
        },
        "feature_derivation": {
            "c2_source_artifact_id": f"{storage_root}/c2/derived.jsonl",
            "logical_artifact_id": "c2-derive-v1",
        },
    }


def _fixture_rows() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    coordinate = {
        "selection_index": 1,
        "selected_example_id": "one",
        "selected_position": 3,
    }
    obligations = [dict(coordinate)]
    passports = [dict(coordinate)]
    payloads = [
        {
            **coordinate,
            "effective_top_k": 1,
            **digest_active_payload_storage(
                {
                    "effective_top_k": 1,
                    "top_token_ids": [7],
                    "top_probs": [0.5],
                    "top_log_probs": [-0.6931471805599453],
                    "top_selection_mask": [True],
                }
            ),
        }
    ]
    return obligations, passports, payloads


def _contract_for_board_summary(board_summary: dict[str, object]) -> dict[str, object]:
    obligations, passports, payloads = _fixture_rows()
    return build_contract(
        fixture_metadata={},
        input_identity={},
        semantic_policy={},
        stage_summary=[],
        selected_obligations=obligations,
        source_passports=passports,
        payload_semantics=payloads,
        board_summary=board_summary,
    )
