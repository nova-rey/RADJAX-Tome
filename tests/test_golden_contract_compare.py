from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import radjax_tome.golden.projection as golden_projection
from radjax_tome.golden.compare import compare_contracts
from radjax_tome.golden.contract import build_contract
from radjax_tome.golden.projection import (
    MAX_GOLDEN_RECORD_BYTES,
    _payload_index,
    _payload_semantics,
    capture_golden_contract,
    validate_fixture,
)
from radjax_tome.golden.projection import (
    _write_fixture as write_captured_fixture,
)
from radjax_tome.tome.golden_fixture import build_production_contract_fixture


def test_compare_allows_one_entropy_quantization_step_but_not_token_drift(
    tmp_path: Path,
) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed", entropy=1.00390625)

    assert compare_contracts(expected, observed)["status"] == "pass"

    payload = observed / "payload_semantics.jsonl"
    row = json.loads(payload.read_text(encoding="utf-8"))
    row["top_token_ids"] = [999]
    payload.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


def test_compare_reports_coordinate_and_selection_order_drift(tmp_path: Path) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed", position=8)

    report = compare_contracts(expected, observed)
    assert report["status"] == "fail"
    assert report["differences"][0]["field"] == "coordinate"


def test_capture_refuses_nonterminal_artifact_without_modifying_it(
    tmp_path: Path,
) -> None:
    artifact = build_production_contract_fixture(tmp_path / "artifact")
    output = tmp_path / "capture"

    with pytest.raises(ValueError, match="production_build_report"):
        capture_golden_contract(artifact, output)

    assert not output.exists()


def test_payload_index_uses_native_selected_exemplars_field(tmp_path: Path) -> None:
    selected = tmp_path / "selected_exemplars"
    selected.mkdir()
    (selected / "payload_index.json").write_text(
        json.dumps(
            {
                "schema_version": "selected_exemplar_payload_index_v1",
                "selected_exemplars": [
                    {"selected_example_id": "one", "selected_position": 3}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _payload_index(tmp_path) == {
        ("one", 3): {"selected_example_id": "one", "selected_position": 3}
    }


def test_capture_fixture_writer_accepts_mkdtemp_staging_directory() -> None:
    staging = Path(tempfile.mkdtemp(prefix="radjax-golden-test-"))
    try:
        fixture = _write_fixture(staging / "contract")
        contract = json.loads((fixture / "contract.json").read_text(encoding="utf-8"))
        write_captured_fixture(
            staging,
            contract,
            _read_jsonl(fixture / "selected_obligations.jsonl"),
            _read_jsonl(fixture / "source_passports.jsonl"),
            _read_jsonl(fixture / "payload_semantics.jsonl"),
            {},
        )

        assert (staging / "contract.json").is_file()
    finally:
        shutil.rmtree(staging)


def test_payload_projection_emits_only_active_padded_entries() -> None:
    projected = _payload_semantics(
        {
            "selected_example_id": "one",
            "selected_position": 3,
            "effective_top_k": 2,
            "top_token_ids": [101, 102, 103, 104],
            "top_probs": [0.4, 0.3, 0.0, 0.0],
            "top_log_probs": [-0.9, -1.2, -100.0, -100.0],
            "top_selection_mask": [True, True, False, False],
            "teacher_entropy": 1.2,
            "top_mass": 0.7,
            "tail_mass": 0.3,
            "long_tail_class": "normal",
            "dynamic_mass_threshold": 0.95,
            "vocab_size": 262144,
            "payload_identity": {"payload_hash": "sha256:payload"},
        },
        1,
    )

    assert projected["top_token_ids"] == [101, 102]
    assert projected["top_probs"] == [0.4, 0.3]
    assert projected["top_log_probs"] == [-0.9, -1.2]
    assert "top_selection_mask" not in projected
    assert all(
        len(projected[key]) == projected["effective_top_k"]
        for key in ("top_token_ids", "top_probs", "top_log_probs")
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("top_selection_mask", [True, False, False, False], "active count"),
        ("top_token_ids", [101, 101, 103, 104], "duplicate active"),
        ("top_probs", [float("nan"), 0.3, 0.0, 0.0], "nonfinite"),
    ],
)
def test_payload_projection_rejects_inconsistent_or_invalid_active_entries(
    field: str,
    value: list[object],
    error: str,
) -> None:
    source: dict[str, object] = {
        "selected_example_id": "one",
        "selected_position": 3,
        "effective_top_k": 2,
        "top_token_ids": [101, 102, 103, 104],
        "top_probs": [0.4, 0.3, 0.0, 0.0],
        "top_log_probs": [-0.9, -1.2, -100.0, -100.0],
        "top_selection_mask": [True, True, False, False],
    }
    source[field] = value

    with pytest.raises(ValueError, match=error):
        _payload_semantics(source, 1)


def test_validate_fixture_rejects_dense_payload_field(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "fixture")
    payload_path = fixture / "payload_semantics.jsonl"
    row = json.loads(payload_path.read_text(encoding="utf-8"))
    row["top_selection_mask"] = [True]
    payload_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden dense payload fields"):
        validate_fixture(fixture)


def test_validate_fixture_rejects_oversized_record(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path / "fixture")
    payload_path = fixture / "payload_semantics.jsonl"
    row = json.loads(payload_path.read_text(encoding="utf-8"))
    row["payload_identity"] = {"padding": "x" * MAX_GOLDEN_RECORD_BYTES}
    payload_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="record exceeds maximum size"):
        validate_fixture(fixture)


def test_compare_streams_jsonl_without_eager_projection_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed")

    def fail_eager_load(_: Path) -> list[dict[str, object]]:
        raise AssertionError("comparison must stream fixture JSONL")

    monkeypatch.setattr(golden_projection, "_read_jsonl", fail_eager_load)

    assert compare_contracts(expected, observed)["status"] == "pass"


def test_capture_projects_payload_shards_without_eager_payload_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    payload_dir = artifact / "selected_exemplars"
    payload_dir.mkdir(parents=True)
    selected = [_selected_record(index) for index in range(1, 257)]
    payload_index = {
        (record["example_id"], record["position"]): {} for record in selected
    }
    _write_payload_shard(payload_dir / "selected-exemplars-00000.json", selected[128:])
    _write_payload_shard(payload_dir / "selected-exemplars-00001.json", selected[:128])

    original_read_object = golden_projection._read_object

    def read_object(path: Path) -> dict[str, object]:
        if not path.is_relative_to(artifact):
            return original_read_object(path)
        if path.name.startswith("selected-exemplars-"):
            return original_read_object(path)
        if path.name == "c6_integrated_selection_validation.json":
            return {"status": "pass"}
        if path.name in {
            "authority_manifest.json",
            "production_build_report.json",
            "validation_report.json",
            "delivery_report.json",
            "selected_linkage_audit.json",
        }:
            return {"status": "pass"}
        raise AssertionError(f"unexpected golden capture read: {path}")

    monkeypatch.setattr(
        golden_projection, "_validate_terminal_artifact", lambda _: None
    )
    monkeypatch.setattr(golden_projection, "_c5_records", lambda _: selected)
    monkeypatch.setattr(golden_projection, "_payload_index", lambda _: payload_index)
    monkeypatch.setattr(golden_projection, "_authority_summary", lambda *_: {})
    monkeypatch.setattr(
        golden_projection,
        "_input_identity",
        lambda _: _valid_input_identity(),
    )
    monkeypatch.setattr(
        golden_projection,
        "_semantic_policy",
        lambda *_: _valid_semantic_policy(),
    )
    monkeypatch.setattr(golden_projection, "_read_object", read_object)
    monkeypatch.setattr(
        golden_projection,
        "_payload_records",
        lambda _: pytest.fail("capture must not use eager payload collection"),
        raising=False,
    )

    capture = tmp_path / "capture"
    report = capture_golden_contract(artifact, capture)

    assert report["status"] == "pass"
    payloads = _read_jsonl(capture / "payload_semantics.jsonl")
    assert [row["selection_index"] for row in payloads] == list(range(1, 257))
    assert all(row["top_token_ids"] == [row["selection_index"]] for row in payloads)
    assert all("top_selection_mask" not in row for row in payloads)


def test_compare_rejects_c5_role_and_passport_drift(tmp_path: Path) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed")

    obligations = observed / "selected_obligations.jsonl"
    row = json.loads(obligations.read_text(encoding="utf-8"))
    row["primary_claim"] = "global"
    obligations.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    [
        ("selected_obligations", "represented_fingerprint_corridor_ids", [99]),
        ("selected_obligations", "selection_obligations", []),
        ("selected_obligations", "selection_index", 8),
        ("payload_semantics", "effective_top_k", 99),
        ("payload_semantics", "bucket_masses", [0.9]),
        ("payload_semantics", "semantic_authority_hash", "sha256:changed"),
    ],
)
def test_compare_rejects_semantic_field_drift(
    tmp_path: Path, collection: str, field: str, value: object
) -> None:
    expected = _write_fixture(tmp_path / "expected")
    observed = _write_fixture(tmp_path / "observed")
    path = observed / f"{collection}.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row[field] = value
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"

    observed = _write_fixture(tmp_path / "passport", position=3)
    passports = observed / "source_passports.jsonl"
    row = json.loads(passports.read_text(encoding="utf-8"))
    row["source_row"] = 99
    passports.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert compare_contracts(expected, observed)["status"] == "fail"


def _write_fixture(root: Path, *, entropy: float = 1.0, position: int = 3) -> Path:
    obligations = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "primary_role": "corridor",
            "primary_claim": "corridor",
            "selection_roles": ["corridor"],
            "selection_obligations": [{"role": "corridor", "rank": 1}],
            "represented_fingerprint_corridor_ids": [7],
            "global_board_ids": ["entropy"],
            "source_passport": {"source_row": 7},
            "payload_identity": {"payload_key": "one:3"},
        }
    ]
    passports = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "source_row": 7,
        }
    ]
    payloads = [
        {
            "selection_index": 1,
            "selected_example_id": "one",
            "selected_position": position,
            "teacher_entropy": entropy,
            "top_token_ids": [7],
            "top_probs": [0.5],
            "top_log_probs": [-0.6931471805599453],
            "effective_top_k": 1,
        }
    ]
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
    root.mkdir()
    (root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    for name, rows in (
        ("selected_obligations", obligations),
        ("source_passports", passports),
        ("payload_semantics", payloads),
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    return root


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _selected_record(index: int) -> dict[str, object]:
    return {
        "example_id": f"example-{index:03d}",
        "position": 0,
        "selection_index": index,
        "source_passport": {"source_row": index - 1},
        "payload_identity": {"payload_key": f"payload-{index:03d}"},
    }


def _write_payload_shard(path: Path, records: list[dict[str, object]]) -> None:
    payloads = [
        {
            "selected_example_id": record["example_id"],
            "selected_position": record["position"],
            "effective_top_k": 1,
            "top_token_ids": [record["selection_index"], 999999],
            "top_probs": [0.9, 0.0],
            "top_log_probs": [-0.1, -100.0],
            "top_selection_mask": [True, False],
            "teacher_entropy": 1.0,
            "top_mass": 0.9,
            "tail_mass": 0.1,
            "dynamic_mass_threshold": 0.95,
            "vocab_size": 262144,
        }
        for record in records
    ]
    path.write_text(json.dumps({"selected_exemplars": payloads}), encoding="utf-8")


def _valid_input_identity() -> dict[str, object]:
    return {
        "teacher_identity": {"model_name": "teacher"},
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


def _valid_semantic_policy() -> dict[str, object]:
    return {
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
