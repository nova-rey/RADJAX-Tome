from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from radjax_tome.golden.contract import (
    build_contract,
    semantic_digest,
    validate_contract,
    validate_sparse_payload_semantics_record,
)

_REQUIRED_REPORTS = (
    "production_build_report.json",
    "validation_report.json",
    "delivery_report.json",
    "selected_linkage_audit.json",
)
MAX_GOLDEN_FIXTURE_BYTES = 64 * 1024 * 1024
MAX_GOLDEN_RECORD_BYTES = 1024 * 1024


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
    input_identity = _input_identity(artifact)
    semantic_policy = _semantic_policy(artifact, reports, authority)
    _require_truth_gate_fields(input_identity, semantic_policy)
    contract = build_contract(
        fixture_metadata={
            "profile": "full_debug_provenance",
            "capture_tool_version": "m2a",
            "repository_revision": reports["production_build_report.json"].get(
                "repository_revision"
            ),
            "canonical_pipeline": "native_two_pass_fingerprint_corridor_path_b",
        },
        input_identity=input_identity,
        semantic_policy=semantic_policy,
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
    validate_contract(contract)
    roots = contract["collection_roots"]
    coordinates: dict[str, set[tuple[str, int]]] = {}
    indices: dict[str, dict[tuple[str, int], int]] = {}
    count = 0
    fixture_bytes = [0]
    for name in ("selected_obligations", "source_passports", "payload_semantics"):
        row_digests: list[str] = []
        seen: set[tuple[str, int]] = set()
        selection_indices: dict[tuple[str, int], int] = {}
        previous_index = -1
        for row in _iter_fixture_jsonl(root / f"{name}.jsonl", fixture_bytes):
            coordinate = _coordinate(row)
            selection_index = row.get("selection_index")
            if coordinate in seen:
                raise ValueError(f"{name} contains duplicate selected coordinates")
            if (
                not isinstance(selection_index, int)
                or isinstance(selection_index, bool)
                or selection_index < 0
                or selection_index <= previous_index
            ):
                raise ValueError(f"{name} selection_index is not strictly ordered")
            if name == "payload_semantics":
                validate_sparse_payload_semantics_record(row)
            seen.add(coordinate)
            selection_indices[coordinate] = selection_index
            previous_index = selection_index
            row_digests.append(semantic_digest(f"{name}-row", row))
        observed_root = semantic_digest(f"{name}-root", row_digests)
        if observed_root != roots[name]:
            raise ValueError(f"golden contract {name} root does not match rows")
        coordinates[name] = seen
        indices[name] = selection_indices
        if name == "selected_obligations":
            count = len(seen)
    selected_coordinates = coordinates["selected_obligations"]
    for name in ("source_passports", "payload_semantics"):
        if coordinates[name] != selected_coordinates:
            raise ValueError(f"selected obligations and {name} do not join")
        if any(
            selection_index != indices["selected_obligations"][coordinate]
            for coordinate, selection_index in indices[name].items()
        ):
            raise ValueError(f"{name} selection_index does not match obligations")
    return {
        "status": "pass",
        "semantic_root": contract["semantic_root"],
        "count": count,
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
    projected = _project(
        row,
        index,
        "effective_top_k",
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
    projected.update(_sparse_payload_arrays(row))
    validate_sparse_payload_semantics_record(projected)
    return projected


def _sparse_payload_arrays(row: dict[str, Any]) -> dict[str, list[Any]]:
    effective_top_k = row.get("effective_top_k")
    arrays = {
        key: row.get(key)
        for key in ("top_token_ids", "top_probs", "top_log_probs", "top_selection_mask")
    }
    if (
        not isinstance(effective_top_k, int)
        or isinstance(effective_top_k, bool)
        or effective_top_k < 1
        or any(not isinstance(value, list) for value in arrays.values())
    ):
        raise ValueError("golden capture payload arrays or effective_top_k are invalid")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("golden capture payload arrays have inconsistent lengths")
    mask = arrays["top_selection_mask"]
    if any(not isinstance(value, bool) for value in mask):
        raise ValueError("golden capture top_selection_mask must contain booleans")
    active_ranks = [rank for rank, active in enumerate(mask) if active]
    if len(active_ranks) != effective_top_k:
        raise ValueError(
            "golden capture top_selection_mask active count does not equal "
            "effective_top_k"
        )
    return {
        key: [arrays[key][rank] for rank in active_ranks]
        for key in ("top_token_ids", "top_probs", "top_log_probs")
    }


def _project(row: dict[str, Any], index: int, *keys: str) -> dict[str, Any]:
    example_id, position = _coordinate(row)
    projected = {
        "selection_index": int(row.get("selection_index", row.get("rank", index))),
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
    run = _read_object(root / "run_manifest.json")
    teacher_manifest = _read_object(root / "teacher_manifest.json")
    emission = _read_object(root / "emission_config.json")
    teacher = teacher_manifest.get("teacher_model_provenance") or emission.get(
        "teacher_model_provenance", {}
    )
    corpus = teacher_manifest.get("corpus_provenance") or emission.get(
        "corpus_provenance", {}
    )
    return {
        "teacher_identity": {
            key: teacher.get(key)
            for key in (
                "model_name",
                "model_revision",
                "config_hash",
                "tokenizer_hash",
                "weights_hash",
                "model_directory_hash",
            )
        },
        "tokenizer_identity": teacher.get("tokenizer_hash"),
        "vocab_size": metadata.get("vocab_size"),
        "sequence_length": metadata.get("sequence_length"),
        "num_examples": metadata.get("num_examples"),
        "teacher_model_hashes": run.get("teacher_model_hashes"),
        "corpus_hash": _first_present(
            corpus,
            run,
            aliases=("source_corpus_hash", "corpus_hash"),
        ),
        "corpus_manifest_hash": _first_present(
            corpus,
            run,
            aliases=("source_corpus_manifest_hash", "manifest_hash"),
        ),
        "normalization_policy": _first_present(
            corpus,
            run,
            aliases=("source_corpus_normalization_policy", "normalization_policy"),
        ),
        "chunking_policy": _first_present(
            corpus,
            run,
            aliases=("source_corpus_chunking_policy", "chunking_policy"),
        ),
        "deduplication_policy": _first_present(
            corpus,
            run,
            aliases=("source_corpus_deduplication_policy", "deduplication_policy"),
        ),
    }


def _semantic_policy(
    root: Path, reports: dict[str, dict[str, Any]], authority: dict[str, Any]
) -> dict[str, Any]:
    emission = _read_object(root / "emission_config.json")
    run = _read_object(root / "run_manifest.json")
    delivery = reports["delivery_report.json"]
    production = reports["production_build_report.json"]
    keys = (
        "teacher_backend",
        "runtime_mode",
        "target_policy",
        "native_execution_mode",
        "selection_integration_policy",
        "dynamic_top_k_min",
        "dynamic_top_k_max",
        "dynamic_mass_threshold",
        "num_buckets",
        "retain_unselected_exemplar_payloads",
    )
    policy = {
        key: _first_present(delivery, production, emission, aliases=(key,))
        for key in keys
    } | {
        "score_pass_config_hash": authority.get("score_pass_config_hash"),
        "selection_integration_config_hash": authority.get(
            "selection_integration_config_hash"
        ),
        "run_manifest_hash": run.get("resume_config_hash"),
    }
    policy["delivery_path"] = _first_present(
        delivery,
        production,
        emission,
        aliases=("delivery_path", "exemplar_delivery_path"),
    )
    policy["selected_rerun_batch_size"] = _first_present(
        delivery,
        production,
        emission,
        aliases=("selected_rerun_requested_batch_size", "selected_rerun_batch_size"),
    )
    return policy


def _authority_summary(root: Path, authority: dict[str, Any]) -> dict[str, Any]:
    required = {
        "c2": root / "c6" / "corridor-leaderboards" / "manifest.json",
        "c3": root / "c6" / "coverage-plan" / "coverage_plan.json",
        "c3_validation": root / "c6" / "coverage-plan" / "validation_report.json",
        "c5": root / "c6" / "multi-role-selection" / "manifest.json",
        "coverage": root / "reports" / "fingerprint_corridor_coverage.json",
        "budget": root / "c6" / "selection_budget_diagnostics.json",
    }
    claim_rows = {
        name: _read_jsonl(root / "c6" / "claims" / name)
        for name in (
            "corridor_claims.jsonl",
            "global_claims.jsonl",
            "collision_obligations.jsonl",
            "selected_coordinates.jsonl",
            "backfill_lineage.jsonl",
        )
    }
    return _semantic_board_summary(
        {
            "authority": _semantic_authority(authority),
            **{name: _read_object(path) for name, path in required.items()},
            "c4_semantic_records": claim_rows,
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
            and key not in {"files", "file_hashes"}
            and not key.endswith("_sha256")
        }
    if isinstance(value, list):
        return [_semantic_board_summary(item) for item in value]
    return value


def _semantic_authority(authority: dict[str, Any]) -> dict[str, Any]:
    """Keep C6 semantic bindings without inheriting C4 storage layout hashes."""
    return {
        key: authority.get(key)
        for key in (
            "schema_version",
            "score_pass_authority_hash",
            "score_pass_config_hash",
            "selection_integration_config_hash",
            "delivery_path",
            "native_execution_mode",
            "corpus_hash",
            "production_grade",
        )
        if authority.get(key) is not None
    }


def _first_present(*sources: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    names = aliases
    for source in sources:
        if not isinstance(source, dict):
            continue
        for name in names:
            if source.get(name) is not None:
                return source[name]
    return None


def _require_truth_gate_fields(
    input_identity: dict[str, Any], semantic_policy: dict[str, Any]
) -> None:
    required_input = (
        "teacher_model_hashes",
        "corpus_hash",
        "corpus_manifest_hash",
        "normalization_policy",
        "chunking_policy",
        "deduplication_policy",
    )
    required_policy = (
        "teacher_backend",
        "runtime_mode",
        "target_policy",
        "native_execution_mode",
        "delivery_path",
        "selection_integration_policy",
        "dynamic_top_k_min",
        "dynamic_top_k_max",
        "dynamic_mass_threshold",
    )
    missing = [
        f"input_identity.{key}"
        for key in required_input
        if input_identity.get(key) is None
    ] + [
        f"semantic_policy.{key}"
        for key in required_policy
        if semantic_policy.get(key) is None
    ]
    teacher_identity = input_identity.get("teacher_identity")
    if (
        not isinstance(teacher_identity, dict)
        or teacher_identity.get("model_name") is None
    ):
        missing.append("input_identity.teacher_identity.model_name")
    teacher_hashes = input_identity.get("teacher_model_hashes")
    if isinstance(teacher_hashes, dict):
        missing.extend(
            f"input_identity.teacher_model_hashes.{key}"
            for key in (
                "config_hash",
                "tokenizer_hash",
                "weights_hash",
                "model_directory_hash",
            )
            if teacher_hashes.get(key) is None
        )
    if missing:
        raise ValueError(
            "golden capture truth gate has required null fields: " + ", ".join(missing)
        )


def _write_fixture(
    root: Path,
    contract: dict[str, Any],
    obligations: list[dict[str, Any]],
    passports: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    board_summary: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
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


def _iter_fixture_jsonl(
    path: Path,
    fixture_bytes: list[int],
) -> Any:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record_bytes = len(line.encode("utf-8"))
            if record_bytes > MAX_GOLDEN_RECORD_BYTES:
                raise ValueError("golden fixture record exceeds maximum size")
            fixture_bytes[0] += record_bytes
            if fixture_bytes[0] > MAX_GOLDEN_FIXTURE_BYTES:
                raise ValueError("golden fixture exceeds maximum size")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("golden fixture JSONL record must be an object")
            yield value
