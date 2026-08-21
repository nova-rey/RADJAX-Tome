"""Finalize a completed M8G selection artifact into a portable replay bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from radjax_contract.tome.workload import (
    SCHEMA_VERSION,
    canonical_json_bytes,
    digest,
    inventory_root,
    validate_checkpoint_manifest,
    validate_finalization_receipt,
    validate_replay_preflight,
    validate_selected_coordinate_inventory,
    validate_selected_source_inventory,
    validate_source_row_closure,
    validate_teacher_inventory,
    validate_workload_authority,
)
from radjax_tome.corpora import validate_corpus_artifact

CORPUS_IDENTITY = (
    "sha256:7719ed62c5bb8feedd7f7e955e52d0b373d8b09e3ec9f6b8256f99a8b5a7e9d1"
)
RAW_INVENTORY_IDENTITY = (
    "sha256:ead1671fcd308f017ce402feafbebea729845d886fe2bef8732aadbf94d9761c"
)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _evidence_sha(root: Path, *relative: str) -> str:
    """Hash the first available durable evidence member, or its absence."""
    for item in relative:
        candidate = root / item
        if candidate.is_file() and not candidate.is_symlink():
            return _sha(candidate)
    return digest({"missing": list(relative)})


def _authority_commits() -> tuple[str, str]:
    """Resolve producer authorities from the checked-out implementation."""
    repo = Path(__file__).resolve().parents[2]
    tome = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"RADJAX-Contract\.git@([0-9a-f]{40})", pyproject)
    if not match:
        raise ValueError("Contract authority pin missing")
    return tome, match.group(1)


def _portable(value: Any, root: str, input_root: str) -> Any:
    if isinstance(value, dict):
        return {k: _portable(v, root, input_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v, root, input_root) for v in value]
    if isinstance(value, str) and (
        os.path.isabs(value) or value.startswith("local:/")
    ):
        if value.startswith("local:/"):
            value = value[len("local:") :]
        for prefix, replacement in (
            (root, "selection-checkpoint"),
            (input_root, "."),
            ("/tmp/m8g-current-1k-build", "selection-checkpoint"),
            ("/inputs/current", "teacher"),
        ):
            if value == prefix or value.startswith(prefix.rstrip("/") + "/"):
                suffix = value[len(prefix) :].lstrip("/")
                return str(PurePosixPath(replacement) / suffix)
        return "provenance://historical/" + Path(value).name
    return value


def _normalize_json_tree(root: Path, raw_root: str, input_root: str) -> None:
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.suffix.lower() not in {".json", ".jsonl"}
        ):
            continue
        # Corpus/model members and raw provenance are byte-authoritative.  Do
        # not reserialize them while projecting producer-local metadata.
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in {"corpus", "model", "raw-provenance"}:
            continue
        try:
            if path.suffix == ".json":
                value = json.loads(path.read_text(encoding="utf-8"))
                path.write_bytes(
                    canonical_json_bytes(_portable(value, raw_root, input_root)) + b"\n"
                )
            else:
                lines = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        lines.append(
                            json.dumps(
                                _portable(json.loads(line), raw_root, input_root),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        )
                path.write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def _inventory(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in bundle: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.startswith("../") or rel.startswith("/"):
            raise ValueError("bundle path escapes root")
        role = (
            "model_member"
            if rel.startswith("model/")
            else "source_row"
            if rel.startswith("source-rows/")
            else "corpus"
            if rel.startswith("corpus/")
            else "checkpoint"
            if rel.startswith("selection-checkpoint/")
            else "provenance"
        )
        entries.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha(path),
                "role": role,
                "semantic_identity": None,
                "schema_profile": None,
                "declaring_record": "workload_authority.json",
                "reason": "portable replay closure",
            }
        )
    return entries


def _project_regular_tree(source: Path, destination: Path) -> None:
    """Project regular files with digest-before-publish semantics."""
    for current, dirs, files in os.walk(source, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"symlink in source tree: {path}")
            (destination / path.relative_to(source)).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = current_path / name
            if src.is_symlink() or not src.is_file():
                raise ValueError(f"non-regular source member: {src}")
            rel = src.relative_to(source)
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            before = src.stat().st_size
            attempt = 0
            while True:
                attempt += 1
                tmp = target.with_name(target.name + f".part-{attempt}")
                try:
                    total = 0
                    h = hashlib.sha256()
                    with src.open("rb") as reader, tmp.open("xb") as writer:
                        while True:
                            block = reader.read(1024 * 1024)
                            if not block:
                                break
                            writer.write(block)
                            h.update(block)
                            total += len(block)
                        writer.flush()
                        os.fsync(writer.fileno())
                    after = src.stat().st_size
                    if total != before or after != before:
                        raise OSError(
                            f"source changed during projection: {src} "
                            f"before={before} read={total} after={after}"
                        )
                    os.replace(tmp, target)
                    break
                except OSError as exc:
                    try:
                        tmp.unlink()
                    except FileNotFoundError:
                        pass
                    if attempt >= 3:
                        raise OSError(f"projection failed for {src}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_workload(
    generation_root: Path, input_root: Path, output: Path
) -> dict[str, Any]:
    generation_root = generation_root.resolve()
    input_root = input_root.resolve()
    output = output.resolve()
    if not generation_root.is_dir() or not input_root.is_dir():
        raise ValueError("generation or input root missing")
    raw_manifest_path = generation_root / "workload_manifest.json"
    if not raw_manifest_path.is_file():
        raise ValueError("raw workload manifest missing")
    raw_manifest = _read_json(raw_manifest_path)
    tome_commit = raw_manifest.get("tome_commit")
    contract_commit = raw_manifest.get("contract_commit")
    if not isinstance(tome_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", tome_commit):
        raise ValueError("raw Tome authority invalid")
    if not isinstance(contract_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", contract_commit):
        raise ValueError("raw Contract authority invalid")
    # The raw producer authority is retained in raw-provenance; finalized
    # workload records bind the reviewed current implementation pin.
    current_tome_commit, current_contract_commit = _authority_commits()
    contract_commit = current_contract_commit
    raw_inventory_identity = raw_manifest.get("file_inventory_digest")
    if not isinstance(raw_inventory_identity, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        raw_inventory_identity
    ):
        raise ValueError("raw workload inventory identity invalid")
    corpus_manifest = _read_json(input_root / "corpus/corpus_manifest.json")
    corpus_identity = corpus_manifest.get("corpus_hash")
    if not isinstance(corpus_identity, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", corpus_identity
    ):
        raise ValueError("corpus identity invalid")
    if output.exists():
        raise FileExistsError(f"conflicting destination exists: {output}")
    checkpoint = generation_root / "selection-checkpoint"
    coord_path = checkpoint / "c6/claims/selected_coordinates.jsonl"
    if not coord_path.is_file():
        raise ValueError("selected coordinate authority missing")
    coords = [json.loads(x) for x in coord_path.read_text().splitlines() if x.strip()]
    if (
        len(coords) != 253
        or len({(x["example_id"], x["position"]) for x in coords}) != 253
    ):
        raise ValueError("frozen coordinate authority is not 253 unique records")
    selected_record_path = (
        checkpoint / "c6/multi-role-selection/selected_exemplars.jsonl"
    )
    selected_records = [
        json.loads(x)
        for x in selected_record_path.read_text().splitlines()
        if x.strip()
    ]
    if len(selected_records) != 253:
        raise ValueError("frozen selected-source record authority is not 253 records")
    selected_record_ids = [r["example_id"] for r in selected_records]
    if len(set(selected_record_ids)) > 253:
        raise ValueError("selected-source record identity is invalid")
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        shutil.rmtree(stage)
        _project_regular_tree(generation_root, stage)
        # Preserve immutable producer provenance byte-for-byte before the
        # portable projection rewrites consumer-facing JSON paths.
        raw_provenance = stage / "raw-provenance"
        raw_provenance.mkdir()
        for name in (
            "teacher_model_provenance.json",
            "runtime_teacher_model_provenance.json",
            "workload_manifest.json",
        ):
            source = stage / name
            if source.is_file():
                shutil.copy2(source, raw_provenance / name)
        # The generation-input provenance is the complete historical teacher
        # record.  Preserve it byte-for-byte; the producer runtime wrapper is
        # not a substitute for this authority-bearing record.
        input_provenance = input_root / "teacher_model_provenance.json"
        if not input_provenance.is_file():
            raise ValueError("complete teacher provenance is missing")
        original_teacher_provenance = input_provenance.read_bytes()
        shutil.copy2(input_provenance, raw_provenance / "teacher_model_provenance.json")
        input_runtime_provenance = input_root / "runtime_teacher_model_provenance.json"
        if not input_runtime_provenance.is_file():
            raise ValueError("complete runtime teacher provenance is missing")
        original_runtime_provenance = input_runtime_provenance.read_bytes()
        (raw_provenance / "runtime_teacher_model_provenance.json").write_bytes(
            original_runtime_provenance
        )
        raw_producer_bytes = {
            name: (raw_provenance / name).read_bytes()
            for name in ("workload_manifest.json",)
            if (raw_provenance / name).is_file()
        }
        # Keep immutable producer records only under raw-provenance. Consumer
        # authority uses the portable records emitted below.
        for name in ("workload_manifest.json", "teacher_model_provenance.json"):
            (stage / name).unlink(missing_ok=True)
        # The verified generation-input closure is the canonical corpus/model
        # authority.  Project it independently so a producer hard-link or
        # stale normalized copy can never contaminate the finalized bundle.
        _project_regular_tree(input_root / "corpus", stage / "corpus")
        _project_regular_tree(input_root / "model", stage / "model")
        corpus_validation = validate_corpus_artifact(stage / "corpus")
        if corpus_validation.status != "pass":
            raise ValueError("canonical corpus validation failed")
        # The corpus JSONL is the verified canonical row source; preserve it as a
        # portable row closure rather than relying on producer-local source paths.
        rows_dir = stage / "source-rows"
        rows_dir.mkdir()
        corpus_rows = []
        source_identity_map = []
        corpus_path = input_root / "corpus/corpus.jsonl"
        for index, line in enumerate(
            corpus_path.read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            source_rel = f"source-{index + 1:04d}.jsonl"
            portable_source_id = f"bundle-source:{row['example_id']}"
            source_identity_map.append(
                {
                    "example_id": row["example_id"],
                    "portable_source_id": portable_source_id,
                    "historical_source_id": row.get("source_id"),
                    "historical_source_path": row.get("source_path"),
                }
            )
            selected = [x for x in coords if x["example_id"] == row.get("example_id")]
            selected_records_for_row = [
                f"selected-record-{i:04d}"
                for i, example_id in enumerate(selected_record_ids)
                if example_id == row.get("example_id")
            ]
            corpus_rows.append(
                {
                    "row_index": index,
                    "example_id": row["example_id"],
                    "source_id": portable_source_id,
                    "source_relative_path": f"source-rows/{source_rel}",
                    "source_file_digest": row.get("source_hash"),
                    "row_digest": digest(row),
                    "corpus_identity": corpus_identity,
                    "selected": bool(selected),
                    "selected_source_records": selected_records_for_row,
                    "selected_coordinates": [
                        {"example_id": x["example_id"], "position": x["position"]}
                        for x in selected
                    ],
                }
            )
            (rows_dir / source_rel).write_text(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n",
                encoding="utf-8",
            )
        if (
            len(corpus_rows) != 1000
            or len({r["example_id"] for r in corpus_rows}) != 1000
        ):
            raise ValueError("corpus does not contain exactly 1000 unique rows")
        closure = rows_dir / "source_row_closure.jsonl"
        closure.write_text(
            "\n".join(
                json.dumps(r, sort_keys=True, separators=(",", ":"))
                for r in corpus_rows
            )
            + "\n"
        )
        validate_source_row_closure(corpus_rows)
        (stage / "raw-provenance/source_identity_provenance.json").write_bytes(
            canonical_json_bytes(
                {
                    "record_type": "source_identity_provenance",
                    "schema_version": SCHEMA_VERSION,
                    "active_source_identity": "bundle-relative-content-bound",
                    "historical_paths_non_authoritative": True,
                    "records": source_identity_map,
                }
            )
            + b"\n"
        )
        (stage / "portable_path_policy.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "active_reference_root": ".",
                    "active_reference_rule": "bundle-relative-only",
                    "historical_provenance_paths_non_authoritative": True,
                    "historical_paths_must_not_be_resolved": True,
                }
            )
            + b"\n"
        )
        _normalize_json_tree(stage, str(generation_root), str(input_root))
        # Raw provenance is immutable evidence, not consumer input.  Restore
        # its exact bytes after portable projection has normalized the staged
        # consumer records.
        (raw_provenance / "teacher_model_provenance.json").write_bytes(
            original_teacher_provenance
        )
        (raw_provenance / "runtime_teacher_model_provenance.json").write_bytes(
            original_runtime_provenance
        )
        for name, raw_bytes in raw_producer_bytes.items():
            (raw_provenance / name).write_bytes(raw_bytes)
        entries = _inventory(stage)
        root = inventory_root(entries)
        (stage / "source_row_closure.jsonl").write_text(closure.read_text())
        closure_digest = _sha(stage / "source_row_closure.jsonl")
        (stage / "source_row_closure.json").write_bytes(
            canonical_json_bytes(
                {
                    "record_type": "source_row_closure",
                    "schema_version": SCHEMA_VERSION,
                    "records": corpus_rows,
                }
            )
            + b"\n"
        )
        (stage / "selected_source_inventory.json").write_bytes(
            canonical_json_bytes(
                {
                    "record_type": "selected_source_inventory",
                    "schema_version": SCHEMA_VERSION,
                    "records": [
                        {
                            "source_id": next(
                                row["source_id"]
                                for row in corpus_rows
                                if row["example_id"] == record["example_id"]
                            ),
                            "example_id": record["example_id"],
                            "selected_source_records": [f"selected-record-{i:04d}"],
                        }
                        for i, record in enumerate(selected_records)
                    ],
                }
            )
            + b"\n"
        )
        (stage / "selected_coordinate_inventory.json").write_bytes(
            canonical_json_bytes(
                {
                    "record_type": "selected_coordinate_inventory",
                    "schema_version": SCHEMA_VERSION,
                    "records": [x for r in corpus_rows for x in r["selected_coordinates"]],
                }
            )
            + b"\n"
        )
        validate_selected_source_inventory(
            json.loads((stage / "selected_source_inventory.json").read_text())
        )
        validate_selected_coordinate_inventory(
            json.loads((stage / "selected_coordinate_inventory.json").read_text())
        )
        teacher = {
            "record_type": "teacher_inventory",
            "schema_version": SCHEMA_VERSION,
            "model_root": "model/model",
            "provenance": "runtime_teacher_model_provenance.json",
            "model_files": [e for e in entries if e["role"] == "model_member"],
            "identity": digest([e for e in entries if e["role"] == "model_member"]),
        }
        validate_teacher_inventory(teacher)
        workload_identity = digest(
            {
                "corpus": corpus_identity,
                "selection": digest(coords),
                "source_rows": closure_digest,
                "teacher": teacher["identity"],
            }
        )
        (stage / "runtime_teacher_model_provenance.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_path": teacher["model_root"],
                    "model_tree_identity": teacher["identity"],
                    "original_provenance": "raw-provenance/teacher_model_provenance.json",
                    "authority_provenance": "runtime_teacher_model_provenance_authority.json",
                    "relocation": "bundle-relative-authority-v1",
                }
            )
            + b"\n"
        )
        projected_provenance = {
            "record_type": "runtime_teacher_provenance",
            "schema_version": SCHEMA_VERSION,
            "model_path": teacher["model_root"],
            "model_tree_identity": teacher["identity"],
            "model_files": teacher["model_files"],
            "original_provenance_digest": _sha(
                stage / "raw-provenance/teacher_model_provenance.json"
            ),
            "relocation": "bundle-relative-authority-v1",
            "runtime_path_authority": "bundle-relative",
        }
        (stage / "runtime_teacher_model_provenance_authority.json").write_bytes(
            canonical_json_bytes(projected_provenance) + b"\n"
        )
        (stage / "teacher_identity.json").write_bytes(
            canonical_json_bytes(teacher) + b"\n"
        )
        validation_summary = {
            "record_type": "workload_validation_report",
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "workload_identity": workload_identity,
            "corpus_identity": corpus_identity,
            "teacher_identity": teacher["identity"],
            "checkpoint_identity": _sha(checkpoint / "c6/authority_manifest.json"),
            "selection_identity": digest(coords),
            "source_row_closure_digest": closure_digest,
            "selected_source_count": 253,
            "selected_coordinate_count": 253,
            "source_count": 1000,
            "example_count": 1000,
            "duplicate_count": 0,
            "underfill_reason": "global_ranked_supply_exhaustion",
            "c1_c5_executed": False,
            "full_teacher_pass_count": 0,
            "materialization_performed": False,
        }
        (stage / "validation_report.json").write_bytes(
            canonical_json_bytes(validation_summary) + b"\n"
        )
        entries = _inventory(stage)
        root = inventory_root(entries)
        checkpoint_manifest = {
            "record_type": "checkpoint_manifest",
            "schema_version": SCHEMA_VERSION,
            "inventory": entries,
            "inventory_root": root,
            "selection_identity": digest(coords),
            "checkpoint_identity": _sha(checkpoint / "c6/authority_manifest.json"),
            "teacher_identity": teacher["identity"],
            "corpus_identity": corpus_identity,
            "selection_config_identity": digest({
                "budget": 256,
                "underfill_reason": "global_ranked_supply_exhaustion",
                "representation_mode": None,
            }),
            "score_pass_identity": _evidence_sha(
                stage,
                "selection-checkpoint/c6/score_pass_manifest.json",
                "selection-checkpoint/c6/authority_manifest.json",
            ),
            "source_row_closure_digest": closure_digest,
            "workload_identity": workload_identity,
            "tome_commit": tome_commit,
            "contract_commit": contract_commit,
        }
        validate_checkpoint_manifest(checkpoint_manifest)
        (stage / "checkpoint_manifest.json").write_bytes(
            canonical_json_bytes(checkpoint_manifest) + b"\n"
        )
        authority = {
            "record_type": "workload_authority",
            "schema_version": SCHEMA_VERSION,
            "workload_identity": workload_identity,
            "tome_commit": tome_commit,
            "contract_commit": contract_commit,
            "corpus_identity": checkpoint_manifest["corpus_identity"],
            "teacher_identity": teacher["identity"],
            "selection_identity": checkpoint_manifest["selection_identity"],
            "selection_policy_identity": digest({
                "corridor": "corridor_first_global_backfill_v1",
                "full_width_cap": {"numerator": 1, "denominator": 3},
                "underfill_reason": "global_ranked_supply_exhaustion",
            }),
            "full_width_cap_policy": {"numerator": 1, "denominator": 3},
            "checkpoint_manifest_digest": _sha(stage / "checkpoint_manifest.json"),
            "source_row_closure_digest": closure_digest,
            "inventory_root": root,
            "replay_identity": digest(
                {
                    "operation": "frozen-selection-replay",
                    "selection": checkpoint_manifest["selection_identity"],
                }
            ),
            "finalization_identity": digest(
                {
                    "inventory": root,
                    "selection": checkpoint_manifest["selection_identity"],
                }
            ),
            "provenance": "NEW_DETERMINISTIC_M8G_1K_WORKLOAD",
            "counts": {
                "sources": 1000,
                "examples": 1000,
                "selected_sources": 253,
                "selected_coordinates": 253,
                "budget": 256,
                "underfill_reason": "global_ranked_supply_exhaustion",
            },
            "source_metadata_policy": {
                "absolute_fields": ["source_id", "source_path", "source_root"],
                "classification": "historical_provenance_only",
                "runtime_resolution_field": "source_relative_path",
                "runtime_resolution_root": "source-rows",
                "absolute_fields_must_not_be_resolved": True,
            },
        }
        validate_workload_authority(authority)
        (stage / "workload_authority.json").write_bytes(
            canonical_json_bytes(authority) + b"\n"
        )
        receipt = {
            "record_type": "finalization_receipt",
            "schema_version": SCHEMA_VERSION,
            "status": "finalized",
            "raw_generation_root_inventory": raw_inventory_identity,
            "original_progress_digest": _evidence_sha(
                generation_root, "selection-checkpoint/production_progress.json"
            ),
            "validation_report_digest": _evidence_sha(
                generation_root,
                "selection-checkpoint/c6/validation_report.json",
                "selection-checkpoint/validation_report.json",
            ),
            "selection_checkpoint_digest": _sha(
                checkpoint / "c6/authority_manifest.json"
            ),
            "checkpoint_manifest_digest": authority["checkpoint_manifest_digest"],
            "source_row_closure_digest": closure_digest,
            "inventory_root": root,
            "finalization_identity": authority["finalization_identity"],
            "tome_commit": tome_commit,
            "contract_commit": contract_commit,
            "configuration_identity": digest({
                "batch_size": 8,
                "sequence_length": 128,
                "budget": 256,
            }),
            "transaction_identity": digest({
                "operation": "finalize-replay-workload",
                "raw_inventory": raw_inventory_identity,
                "output": "portable-authority-bundle",
            }),
            "benchmark_performed": False,
            "materialization_performed": False,
        }
        validate_finalization_receipt(receipt)
        (stage / "finalization_receipt.json").write_bytes(
            canonical_json_bytes(receipt) + b"\n"
        )
        # Preflight records are authority evidence only: no teacher, GPU, or
        # representation materialization is invoked by this finalizer.
        for mode in (
            "legacy_padded_monolithic",
            "compact_k_monolithic",
            "compact_k_immutable_body",
        ):
            preflight = {
                "record_type": "replay_preflight",
                "schema_version": SCHEMA_VERSION,
                "mode": mode,
                "requested_mode": mode,
                "executed_mode": mode,
                "status": "pass",
                "workload_identity": authority["workload_identity"],
                "selection_identity": authority["selection_identity"],
                "selected_coordinate_identity": authority["selection_identity"],
                "selected_sources": 253,
                "selected_coordinates": 253,
                "c1_c5_skipped": True,
                "full_teacher_pass_count": 0,
                "gpu_requested": False,
                "fallback": False,
                "selected_delivery_status": "not_started",
                "materialization_performed": False,
                "publication_performed": False,
                "resume_identity": digest({
                    "workload": authority["workload_identity"],
                    "mode": mode,
                }),
            }
            validate_replay_preflight(preflight)
            (stage / f"replay_preflight_{mode}.json").write_bytes(
                canonical_json_bytes(preflight) + b"\n"
            )
        (stage / "M8D_COMPARABILITY.md").write_text(
            "M8D used the historical 213-source/256-coordinate workload. "
            "This new deterministic M8G 1K workload contains 1,000 sources "
            "and 253 selected coordinates. Absolute values are contextual "
            "only; future three-mode comparisons are internally paired on "
            "this exact bundle. No percentage threshold is used.\n"
        )
        final_entries = _inventory(stage)
        final_root = inventory_root(final_entries)
        (stage / "bundle_inventory.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": SCHEMA_VERSION,
                    "entries": final_entries,
                    "inventory_root": final_root,
                }
            )
            + b"\n"
        )
        os.replace(stage, output)
        return {
            "status": "pass",
            "output": str(output),
            "inventory_root": final_root,
            "workload_identity": authority["workload_identity"],
            "counts": authority["counts"],
        }
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
