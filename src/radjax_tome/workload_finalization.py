"""Finalize a completed M8G selection artifact into a portable replay bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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
    validate_source_row_closure,
    validate_teacher_inventory,
    validate_workload_authority,
)

CORPUS_IDENTITY = (
    "sha256:7719ed62c5bb8feedd7f7e955e52d0b373d8b09e3ec9f6b8256f99a8b5a7e9d1"
)
RAW_INVENTORY_IDENTITY = (
    "sha256:ead1671fcd308f017ce402eafbebea729845d886fe2bef8732aadbf94d9761c"
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


def _portable(value: Any, root: str, input_root: str) -> Any:
    if isinstance(value, dict):
        return {k: _portable(v, root, input_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v, root, input_root) for v in value]
    if isinstance(value, str) and os.path.isabs(value):
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
            try:
                os.link(src, target)
                continue
            except OSError:
                pass
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
        # The corpus JSONL is the verified canonical row source; preserve it as a
        # portable row closure rather than relying on producer-local source paths.
        rows_dir = stage / "source-rows"
        rows_dir.mkdir()
        corpus_rows = []
        corpus_path = input_root / "corpus/corpus.jsonl"
        for index, line in enumerate(
            corpus_path.read_text(encoding="utf-8").splitlines()
        ):
            row = json.loads(line)
            source_rel = f"source-{index + 1:04d}.jsonl"
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
                    "source_id": row["source_id"],
                    "source_relative_path": f"source-rows/{source_rel}",
                    "source_file_digest": row.get("source_hash"),
                    "row_digest": digest(row),
                    "corpus_identity": CORPUS_IDENTITY,
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
        _normalize_json_tree(stage, str(generation_root), str(input_root))
        entries = _inventory(stage)
        root = inventory_root(entries)
        (stage / "source_row_closure.jsonl").write_text(closure.read_text())
        closure_digest = _sha(stage / "source_row_closure.jsonl")
        teacher = {
            "schema_version": SCHEMA_VERSION,
            "model_root": "model/model",
            "provenance": "runtime_teacher_model_provenance.json",
            "model_files": [e for e in entries if e["role"] == "model_member"],
            "identity": digest([e for e in entries if e["role"] == "model_member"]),
        }
        validate_teacher_inventory(teacher)
        (stage / "teacher_identity.json").write_bytes(
            canonical_json_bytes(teacher) + b"\n"
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
            "corpus_identity": CORPUS_IDENTITY,
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
            "workload_identity": digest({
                "corpus": CORPUS_IDENTITY,
                "selection": digest(coords),
                "source_rows": closure_digest,
            }),
        }
        validate_checkpoint_manifest(checkpoint_manifest)
        (stage / "checkpoint_manifest.json").write_bytes(
            canonical_json_bytes(checkpoint_manifest) + b"\n"
        )
        authority = {
            "record_type": "workload_authority",
            "schema_version": SCHEMA_VERSION,
            "workload_identity": digest(
                {
                    "coords": coords,
                    "corpus": checkpoint_manifest["corpus_identity"],
                    "teacher": teacher["identity"],
                }
            ),
            "tome_commit": "b350b0b89e97766336e9b2e64d9dbe9e1c4a712e",
            "contract_commit": "7c2e394be5a2848ef157c9de02f8edad9fb25b72",
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
        }
        validate_workload_authority(authority)
        (stage / "workload_authority.json").write_bytes(
            canonical_json_bytes(authority) + b"\n"
        )
        receipt = {
            "record_type": "finalization_receipt",
            "schema_version": SCHEMA_VERSION,
            "status": "finalized",
            "raw_generation_root_inventory": RAW_INVENTORY_IDENTITY,
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
            "tome_commit": "b350b0b89e97766336e9b2e64d9dbe9e1c4a712e",
            "contract_commit": "7c2e394be5a2848ef157c9de02f8edad9fb25b72",
            "configuration_identity": digest({
                "batch_size": 8,
                "sequence_length": 128,
                "budget": 256,
            }),
            "transaction_identity": digest({
                "operation": "finalize-replay-workload",
                "raw_inventory": RAW_INVENTORY_IDENTITY,
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
