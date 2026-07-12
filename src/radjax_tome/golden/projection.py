from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from radjax_tome.golden.contract import build_contract, validate_contract

_REQUIRED_REPORTS = (
    "production_build_report.json",
    "validation_report.json",
    "delivery_report.json",
    "selected_linkage_audit.json",
)


def capture_golden_contract(artifact_dir: Path, output_dir: Path) -> dict[str, Any]:
    artifact = artifact_dir.resolve()
    _validate_terminal_artifact(artifact)
    selected = _c5_records(artifact)
    payload_index = _payload_index(artifact)
    payloads = _payload_records(artifact)
    if len(selected) != 256 or len(payloads) != 256 or len(payload_index) != 256:
        raise ValueError("golden capture requires exactly 256 selected coordinates")
    _require_coordinate_join(selected, payload_index, payloads)
    obligations = [
        _obligation(row, index) for index, row in enumerate(selected, start=1)
    ]
    passports = [_passport(row, index) for index, row in enumerate(selected, start=1)]
    payload_by_coordinate = {_coordinate(row): row for row in payloads}
    payload_rows = [
        _payload_semantics(
            {
                **payload_by_coordinate[_coordinate(record)],
                "selection_index": record["selection_index"],
                "payload_identity": record["payload_identity"],
            },
            index,
        )
        for index, record in enumerate(selected, start=1)
    ]
    authority = _read_object(artifact / "c6" / "authority_manifest.json")
    board_summary = _authority_summary(artifact, authority)
    reports = {name: _read_object(artifact / name) for name in _REQUIRED_REPORTS}
    reconciliation = _read_object(
        artifact / "reports" / "c6_integrated_selection_validation.json"
    )
    if reconciliation.get("status") != "pass":
        raise ValueError("golden capture requires passing C6 selection reconciliation")
    contract = build_contract(
        fixture_metadata={
            "profile": "full_debug_provenance",
            "capture_tool_version": "m2a",
            "repository_revision": reports["production_build_report.json"].get(
                "repository_revision"
            ),
            "canonical_pipeline": "native_two_pass_fingerprint_corridor_path_b",
        },
        input_identity=_input_identity(artifact),
        semantic_policy=_semantic_policy(artifact, reports, authority),
        stage_summary=[
            {"stage": name.removesuffix(".json"), "status": report.get("status")}
            for name, report in reports.items()
        ]
        + [{"stage": "c6_reconciliation", "status": reconciliation["status"]}],
        selected_obligations=obligations,
        source_passports=passports,
        payload_semantics=payload_rows,
        board_summary=board_summary,
        capture_metadata={"capture_status": "captured", "source_path": "redacted"},
    )
    temporary = Path(tempfile.mkdtemp(prefix="radjax-golden-", dir=output_dir.parent))
    try:
        _write_fixture(
            temporary, contract, obligations, passports, payload_rows, board_summary
        )
        validate_fixture(temporary)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "pass", "semantic_root": contract["semantic_root"], "count": 256}


def validate_fixture(fixture_dir: Path) -> dict[str, Any]:
    root = fixture_dir.resolve()
    contract = _read_object(root / "contract.json")
    collections = {
        "selected_obligations": _read_jsonl(root / "selected_obligations.jsonl"),
        "source_passports": _read_jsonl(root / "source_passports.jsonl"),
        "payload_semantics": _read_jsonl(root / "payload_semantics.jsonl"),
    }
    validate_contract(contract, collections=collections)
    return {
        "status": "pass",
        "semantic_root": contract["semantic_root"],
        "count": len(collections["selected_obligations"]),
    }


def _validate_terminal_artifact(root: Path) -> None:
    for name in _REQUIRED_REPORTS:
        report = _read_object(root / name)
        if report.get("status") != "pass" or report.get("blockers"):
            raise ValueError(f"golden capture requires terminal pass report: {name}")
    _payload_index(root)
    reconciliation = _read_object(
        root / "reports" / "c6_integrated_selection_validation.json"
    )
    if reconciliation.get("status") != "pass":
        raise ValueError("golden capture requires passing C6 selection reconciliation")


def _c5_records(root: Path) -> list[dict[str, Any]]:
    path = root / "c6" / "multi-role-selection" / "selected_exemplars.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("golden capture C5 rich selected records are missing")
    return rows


def _payload_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    index = _read_object(root / "selected_exemplars" / "payload_index.json")
    rows = index.get("selected_exemplars")
    if not isinstance(rows, list):
        raise ValueError("golden capture payload index selected_exemplars are missing")
    result = {_coordinate(row): dict(row) for row in rows if isinstance(row, dict)}
    if len(result) != len(rows):
        raise ValueError("golden capture payload index has duplicate coordinates")
    return result


def _payload_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "selected_exemplars").glob("selected-exemplars-*.json")):
        document = _read_object(path)
        values = document.get("selected_exemplars", document.get("payloads"))
        if isinstance(values, list):
            rows.extend(dict(row) for row in values if isinstance(row, dict))
    return rows


def _require_coordinate_join(
    selected: list[dict[str, Any]],
    payload_index: dict[tuple[str, int], dict[str, Any]],
    payloads: list[dict[str, Any]],
) -> None:
    c5 = {_coordinate(row) for row in selected}
    payload = {_coordinate(row) for row in payloads}
    if len(c5) != len(selected) or c5 != set(payload_index) or c5 != payload:
        raise ValueError(
            "golden capture C5, payload index, and payload coordinates do not join"
        )


def _obligation(row: dict[str, Any], index: int) -> dict[str, Any]:
    projected = _project(
        row,
        index,
        "primary_role",
        "primary_claim",
        "selection_roles",
        "selection_obligations",
        "represented_fingerprint_corridor_ids",
        "global_board_ids",
        "source_passport",
        "payload_identity",
    )
    projected["selected_example_id"] = str(row["example_id"])
    projected["selected_position"] = int(row["position"])
    return projected


def _passport(row: dict[str, Any], index: int) -> dict[str, Any]:
    passport = dict(row.get("source_passport") or {})
    passport.update(
        {
            "selection_index": int(row["selection_index"]),
            "selected_example_id": str(row["example_id"]),
            "selected_position": int(row["position"]),
        }
    )
    return passport


def _payload_semantics(row: dict[str, Any], index: int) -> dict[str, Any]:
    return _project(
        row,
        index,
        "effective_top_k",
        "top_token_ids",
        "top_probs",
        "top_log_probs",
        "top_selection_mask",
        "bucket_masses",
        "teacher_entropy",
        "top_mass",
        "tail_mass",
        "long_tail_class",
        "vocab_size",
        "dynamic_mass_threshold",
        "source_policy",
        "semantic_authority_hash",
        "payload_identity",
    )


def _project(row: dict[str, Any], index: int, *keys: str) -> dict[str, Any]:
    example_id, position = _coordinate(row)
    projected = {
        "selection_index": int(row.get("rank", row.get("selection_index", index))),
        "selected_example_id": example_id,
        "selected_position": position,
    }
    for key in keys:
        if key in row:
            projected[key] = row[key]
    return projected


def _coordinate(row: dict[str, Any]) -> tuple[str, int]:
    example_id = row.get("selected_example_id", row.get("example_id"))
    position = row.get("selected_position", row.get("position"))
    if not isinstance(example_id, str) or not isinstance(position, int):
        raise ValueError("golden capture record has invalid coordinate")
    return example_id, position


def _input_identity(root: Path) -> dict[str, Any]:
    metadata = _read_object(root / "metadata.json")
    return {
        key: metadata.get(key)
        for key in (
            "model_id",
            "tokenizer_id",
            "vocab_size",
            "sequence_length",
            "num_examples",
        )
    }


def _semantic_policy(
    root: Path, reports: dict[str, dict[str, Any]], authority: dict[str, Any]
) -> dict[str, Any]:
    emission = _read_object(root / "emission_config.json")
    run = _read_object(root / "run_manifest.json")
    keys = (
        "teacher_backend",
        "runtime_mode",
        "target_policy",
        "native_execution_mode",
        "selection_integration_policy",
        "exemplar_delivery_path",
        "dynamic_top_k_min",
        "dynamic_top_k_max",
        "dynamic_mass_threshold",
        "num_buckets",
        "selected_rerun_batch_size",
        "retain_unselected_exemplar_payloads",
    )
    return {
        key: emission.get(key, reports["delivery_report.json"].get(key)) for key in keys
    } | {
        "score_pass_config_hash": authority.get("score_pass_config_hash"),
        "selection_integration_config_hash": authority.get(
            "selection_integration_config_hash"
        ),
        "run_manifest_hash": run.get("resume_config_hash"),
    }


def _authority_summary(root: Path, authority: dict[str, Any]) -> dict[str, Any]:
    paths = authority.get("paths") or {}
    required = {
        "c2": root / "c6" / "corridor-leaderboards" / "manifest.json",
        "c3": root / "c6" / "coverage-plan" / "manifest.json",
        "c4": root / "c6" / "claims" / "claim_manifest.json",
        "c5": root / "c6" / "multi-role-selection" / "manifest.json",
        "coverage": root / "reports" / "fingerprint_corridor_coverage.json",
        "budget": root / "c6" / "selection_budget_diagnostics.json",
    }
    return _semantic_board_summary(
        {
            "authority": authority,
            "paths": paths,
            **{name: _read_object(path) for name, path in required.items()},
        }
    )


def _semantic_board_summary(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_board_summary(item)
            for key, item in value.items()
            if key
            not in {
                "created_at",
                "elapsed_seconds",
                "host_memory_bytes",
                "device_memory_bytes",
                "path",
                "shard_path",
            }
            and not key.endswith("_path")
        }
    if isinstance(value, list):
        return [_semantic_board_summary(item) for item in value]
    return value


def _write_fixture(
    root: Path,
    contract: dict[str, Any],
    obligations: list[dict[str, Any]],
    passports: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    board_summary: dict[str, Any],
) -> None:
    root.mkdir(parents=True)
    (root / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    (root / "board_summary.json").write_text(
        json.dumps(board_summary, indent=2, sort_keys=True) + "\n"
    )
    for name, rows in (
        ("selected_obligations", obligations),
        ("source_passports", passports),
        ("payload_semantics", payloads),
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"golden capture missing required file: {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"golden capture expected object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
