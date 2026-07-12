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
    selected = _selected_records(artifact)
    payloads = _payload_records(artifact)
    if len(selected) != 256 or len(payloads) != 256:
        raise ValueError("golden capture requires exactly 256 selected coordinates")
    obligations = [
        _obligation(row, index) for index, row in enumerate(selected, start=1)
    ]
    passports = [_passport(row, index) for index, row in enumerate(selected, start=1)]
    payload_rows = [
        _payload_semantics(row, index) for index, row in enumerate(payloads, start=1)
    ]
    board_summary = _semantic_board_summary(
        _read_object(artifact / "leaderboards" / "leaderboard_report.json")
    )
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
        semantic_policy=_semantic_policy(reports),
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
    index = _read_object(root / "selected_exemplars" / "payload_index.json")
    if not isinstance(index.get("records"), list):
        raise ValueError("golden capture requires selected payload index records")
    reconciliation = _read_object(
        root / "reports" / "c6_integrated_selection_validation.json"
    )
    if reconciliation.get("status") != "pass":
        raise ValueError("golden capture requires passing C6 selection reconciliation")


def _selected_records(root: Path) -> list[dict[str, Any]]:
    document = _read_object(root / "leaderboards" / "selected_exemplars.json")
    records = document.get("selected_exemplars")
    if not isinstance(records, list):
        raise ValueError("golden capture selected_exemplars records are missing")
    return [dict(row) for row in records if isinstance(row, dict)]


def _payload_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "selected_exemplars").glob("selected-exemplars-*.json")):
        document = _read_object(path)
        values = document.get("selected_exemplars", document.get("payloads"))
        if isinstance(values, list):
            rows.extend(dict(row) for row in values if isinstance(row, dict))
    return rows


def _obligation(row: dict[str, Any], index: int) -> dict[str, Any]:
    return _project(
        row,
        index,
        "selected_board",
        "primary_role",
        "fulfilled_roles",
        "corridor_mode_id",
        "corridor_fingerprint_id",
        "source_shard_id",
        "source_row",
        "source_score",
        "source_top_token_id",
        "score_pass_authority_hash",
        "delivery_authority_hash",
    )


def _passport(row: dict[str, Any], index: int) -> dict[str, Any]:
    return _project(
        row,
        index,
        "source_shard_id",
        "source_row",
        "source_position",
        "source_score",
        "source_top_token_id",
        "source_score_policy",
        "payload_ref",
        "score_pass_authority_hash",
        "delivery_authority_hash",
    )


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
    )


def _project(row: dict[str, Any], index: int, *keys: str) -> dict[str, Any]:
    projected = {
        "selection_index": int(row.get("rank", row.get("selection_index", index))),
        "selected_example_id": row["selected_example_id"],
        "selected_position": int(row["selected_position"]),
    }
    for key in keys:
        if key in row:
            projected[key] = row[key]
    return projected


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


def _semantic_policy(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    delivery = reports["delivery_report.json"]
    return {
        key: delivery.get(key)
        for key in (
            "delivery_path",
            "dynamic_top_k_min",
            "dynamic_top_k_max",
            "dynamic_mass_threshold",
            "num_buckets",
            "selection_integration_policy",
        )
    }


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
