"""Strict adoption of an already-verified frozen selected-source authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.c6_integration import c5_records_for_delivery
from radjax_tome.corpora import validate_corpus_artifact
from radjax_tome.fingerprint.multi_role_selection import (
    load_multi_role_selection_artifact,
    load_multi_role_selection_artifact_for_replay,
)
from radjax_tome.provenance.teacher_model import (
    inspect_teacher_model,
    validate_teacher_model_provenance,
)


@dataclass(frozen=True)
class FrozenSelectionReplay:
    replay_root: Path
    bundle_manifest: Path
    adopted_root: Path
    bundle_manifest_sha256: str
    checkpoint_digest: str
    selected_record_digest: str
    replay_identity: str
    records: tuple[dict[str, Any], ...]
    selected_sources: int
    selected_coordinates: int
    metadata_digest: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _publish_replay_metadata(
    *, source: Path, run_root: Path, expected_digest: str
) -> None:
    """Atomically expose authority-bound score-pass metadata at RUN_ROOT."""
    if not source.is_file() or source.is_symlink():
        raise ValueError("frozen replay metadata is not a regular file")
    actual = _sha256(source)
    if actual != expected_digest:
        raise ValueError("frozen replay metadata digest mismatch")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("frozen replay metadata is not valid JSON") from exc
    if payload.get("schema_version") != "qrwkv_xla.teacher_target_store.v1":
        raise ValueError("frozen replay metadata schema is unsupported")
    run_root.mkdir(parents=True, exist_ok=True)
    destination = run_root / "metadata.json"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("run-root metadata destination has the wrong type")
        if _sha256(destination) != expected_digest:
            raise ValueError("run-root metadata destination conflicts with authority")
        return
    fd, temporary = tempfile.mkstemp(prefix=".metadata.", dir=run_root)
    try:
        with os.fdopen(fd, "wb") as handle, source.open("rb") as source_handle:
            shutil.copyfileobj(source_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path = Path(temporary)
        if _sha256(temporary_path) != expected_digest:
            raise ValueError("staged replay metadata digest mismatch")
        os.replace(temporary_path, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _publish_replay_shards(*, source: Path, run_root: Path) -> str:
    """Atomically expose the adopted score-pass shard closure at RUN_ROOT."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("frozen replay shard closure is not a regular directory")
    for path in source.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ValueError("frozen replay shard closure contains unsafe member")
    destination = run_root / "shards"
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("run-root shard destination has the wrong type")
        source_files = sorted(source.rglob("*.npz"))
        dest_files = sorted(destination.rglob("*.npz"))
        if [p.name for p in source_files] != [p.name for p in dest_files]:
            raise ValueError("run-root shard destination conflicts with authority")
        for left, right in zip(source_files, dest_files, strict=True):
            if _sha256(left) != _sha256(right):
                raise ValueError("run-root shard destination digest conflict")
    else:
        temporary = run_root / ".shards.staging"
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary, symlinks=False)
        os.replace(temporary, destination)
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [
                    {"path": p.relative_to(run_root).as_posix(), "sha256": _sha256(p)}
                    for p in sorted(destination.rglob("*"))
                    if p.is_file()
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def _owned_member_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("private replay member path must be text")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("private replay member path escapes adoption root")
    path = root / candidate
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("private replay member path escapes adoption root") from exc
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("private replay member path contains a symlink")
    return path


def _replay_identity(
    *,
    bundle_manifest_sha256: str,
    checkpoint_digest: str,
    selected_record_digest: str,
    selected_sources: int = 213,
    selected_coordinates: int = 256,
) -> str:
    payload = {
        "operation": "frozen_selection_replay_v1",
        "bundle_manifest_sha256": bundle_manifest_sha256,
        "checkpoint_digest": checkpoint_digest,
        "selected_record_digest": selected_record_digest,
        "selected_sources": selected_sources,
        "selected_coordinates": selected_coordinates,
    }
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _load_validator():
    script = (
        Path(__file__).resolve().parents[4]
        / "scripts"
        / "validate_m8g_workload_bundle.py"
    )
    if not script.is_file():
        raise ValueError("frozen replay validator is unavailable")
    spec = importlib.util.spec_from_file_location("_m8g_workload_validator", script)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load frozen replay validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adopt_verified_selection_replay(
    *,
    bundle_manifest: Path,
    artifact_root: Path,
    adopted_root: Path,
) -> FrozenSelectionReplay:
    """Validate and privately adopt a frozen authority before any inference."""

    bundle_manifest = bundle_manifest.expanduser()
    artifact_root = artifact_root.expanduser()
    adopted_root = adopted_root.expanduser()
    if bundle_manifest.is_symlink() or not bundle_manifest.is_file():
        raise ValueError("frozen replay manifest must be a regular file")
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise ValueError("frozen replay artifact root must be a regular directory")
    if adopted_root.exists() and adopted_root.is_symlink():
        raise ValueError("private replay adoption root must not be a symlink")
    bundle_manifest = bundle_manifest.resolve()
    artifact_root = artifact_root.resolve()
    adopted_root = adopted_root.resolve()
    document = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    # Current finalized M8G bundles are self-describing authority roots.  They
    # do not use the historical validator's 213/256 contract or layout.
    authority_path = artifact_root / "workload_authority.json"
    if authority_path.is_file() and not authority_path.is_symlink():
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        try:
            from radjax_contract.tome import validate_role_binding

            validate_role_binding("workload_authority", "authority")
            validate_role_binding(
                "runtime_teacher_provenance", "runtime_teacher_provenance"
            )
        except ImportError as exc:
            raise ValueError("replay Contract role-binding API unavailable") from exc
        if authority.get("provenance") != "NEW_DETERMINISTIC_M8G_1K_WORKLOAD":
            raise ValueError("current replay workload provenance invalid")
        runtime_locator = artifact_root / "runtime_teacher_model_provenance.json"
        if not runtime_locator.is_file() or runtime_locator.is_symlink():
            raise ValueError("current replay runtime teacher provenance is missing")
        runtime_record = json.loads(runtime_locator.read_text(encoding="utf-8"))
        if runtime_record.get("schema_version") != "teacher_model_provenance_v1":
            if (
                authority.get("replay_compatibility_version")
                != "current-production-replay-v1"
            ):
                raise ValueError(
                    "teacher_model_provenance.json schema_version is unsupported"
                )
            raise ValueError(
                "current replay runtime teacher provenance must use "
                "teacher_model_provenance_v1"
            )
        selected_sources = int(authority["counts"]["selected_sources"])
        selected_coordinates = int(authority["counts"]["selected_coordinates"])
        replay_root = artifact_root
        metadata_source = replay_root / "selection-checkpoint/metadata.json"
        if not metadata_source.is_file() or metadata_source.is_symlink():
            raise ValueError("current replay metadata record is missing")
        metadata_digest = _sha256(metadata_source)
        inventory_entries = document.get("entries") or document.get("inventory") or []
        metadata_entries = [
            entry
            for entry in inventory_entries
            if entry.get("path") == "selection-checkpoint/metadata.json"
        ]
        if len(metadata_entries) != 1:
            raise ValueError("current replay metadata role is not uniquely bound")
        expected_metadata_digest = metadata_entries[0].get("sha256")
        if expected_metadata_digest != metadata_digest:
            raise ValueError("current replay metadata authority digest mismatch")
        coord_file = (
            replay_root / "selection-checkpoint/c6/claims/selected_coordinates.jsonl"
        )
        record_file = (
            replay_root
            / "selection-checkpoint/c6/multi-role-selection/selected_exemplars.jsonl"
        )
        if not coord_file.is_file() or not record_file.is_file():
            raise ValueError("current replay selection records are missing")
        policy = json.loads((replay_root / "portable_path_policy.json").read_text())
        if policy.get(
            "active_reference_rule"
        ) != "bundle-relative-only" or not policy.get(
            "historical_paths_must_not_be_resolved"
        ):
            raise ValueError("current replay path policy is not closed")
        if validate_corpus_artifact(replay_root / "corpus").status != "pass":
            raise ValueError("current replay corpus closure is invalid")
        authority_provenance = (
            replay_root / "runtime_teacher_model_provenance_authority.json"
        )
        teacher_identity = replay_root / "teacher_identity.json"
        if not authority_provenance.is_file() or not teacher_identity.is_file():
            raise ValueError("current replay teacher closure is incomplete")
        coords = [
            json.loads(x) for x in coord_file.read_text().splitlines() if x.strip()
        ]
        raw_records = [
            json.loads(x) for x in record_file.read_text().splitlines() if x.strip()
        ]
        if len(coords) != selected_coordinates or len(raw_records) != selected_sources:
            raise ValueError("current replay selection counts mismatch")
        selected_record_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(raw_records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        # Reuse the canonical C5-to-delivery translation used by ordinary
        # production.  The raw multi-role records keep selection authority;
        # delivery records bind source_shard_id/source_row to the passport.
        selected_artifact = load_multi_role_selection_artifact_for_replay(
            replay_root / "selection-checkpoint/c6/multi-role-selection"
        )
        records = c5_records_for_delivery(
            selected_artifact, delivery_path="two_pass_rerun_selected"
        )
        if len(records) != selected_sources:
            raise ValueError("current replay delivery record count mismatch")
        checkpoint_digest = str(authority["checkpoint_manifest_digest"])
        bundle_manifest_sha256 = _sha256(bundle_manifest)
        replay_identity = _replay_identity(
            bundle_manifest_sha256=bundle_manifest_sha256,
            checkpoint_digest=checkpoint_digest,
            selected_record_digest=selected_record_digest,
            selected_sources=selected_sources,
            selected_coordinates=selected_coordinates,
        )
        if adopted_root.exists():
            if not (adopted_root / "replay_authority.json").is_file():
                raise ValueError("private replay adoption is incomplete")
            prior = json.loads((adopted_root / "replay_authority.json").read_text())
            if prior.get("replay_identity") != replay_identity:
                raise ValueError(
                    "private replay adoption conflicts with current authority"
                )
            expected_closure = prior.get("input_closure") or {}
            actual_closure = {}
            for member in sorted((adopted_root / "input").rglob("*")):
                if member.is_symlink():
                    raise ValueError("current replay adopted input contains symlink")
                if member.is_dir():
                    continue
                if not member.is_file():
                    raise ValueError(
                        "current replay adopted input contains special file"
                    )
                actual_closure[member.relative_to(adopted_root).as_posix()] = _sha256(
                    member
                )
            if set(actual_closure) != set(expected_closure):
                raise ValueError("current replay input closure member set changed")
            for relative, expected_digest in expected_closure.items():
                member = _owned_member_path(adopted_root, relative)
                if (
                    member.is_symlink()
                    or not member.is_file()
                    or _sha256(member) != expected_digest
                ):
                    raise ValueError(
                        f"current replay input closure mismatch: {relative}"
                    )
        else:
            adopted_root.mkdir(parents=True, exist_ok=False)
            # Canonical production preflight consumes an invocation-owned
            # input layout.  Adopt the complete verified closure, not just a
            # marker, and record every copied member digest.
            input_root = adopted_root / "input"
            for relative in ("corpus", "model", "source-rows"):
                source_tree = replay_root / relative
                if any(path.is_symlink() for path in source_tree.rglob("*")):
                    raise ValueError(
                        f"current replay closure contains symlink: {relative}"
                    )
                shutil.copytree(source_tree, input_root / relative, symlinks=False)
            shard_tree = replay_root / "selection-checkpoint/shards"
            if not shard_tree.is_dir() or shard_tree.is_symlink():
                raise ValueError("current replay score-pass shard closure is missing")
            if any(path.is_symlink() for path in shard_tree.rglob("*")):
                raise ValueError(
                    "current replay score-pass shard closure contains symlink"
                )
            shutil.copytree(shard_tree, input_root / "shards", symlinks=False)
            for relative in (
                "runtime_teacher_model_provenance_authority.json",
                "teacher_identity.json",
                "runtime_teacher_model_provenance.json",
                "portable_path_policy.json",
            ):
                target = input_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(replay_root / relative, target)
            # Metadata is a checkpoint-bound production input.  Adopt it
            # under a stable input role; production publishes it atomically
            # at RUN_ROOT before selected delivery.
            shutil.copy2(metadata_source, input_root / "metadata.json")
            for source_name, target_name in (
                ("corpus/corpus.jsonl", "corpus.jsonl"),
                ("corpus/corpus_manifest.json", "corpus_manifest.json"),
                ("corpus/corpus_build_report.json", "corpus_build_report.json"),
            ):
                shutil.copy2(input_root / source_name, input_root / target_name)

            # The workload-level runtime locator is provenance, not the
            # teacher_model_provenance_v1 record consumed by production.
            # Derive a private v1 projection only after the complete adopted
            # model tree exists, bind it to the immutable model authority, and
            # validate it through the exact production validator.
            model_root = input_root / "model" / "model"
            projection = inspect_teacher_model(model_root, check="metadata_only")
            projection["portable_source"] = (
                "runtime_teacher_model_provenance_authority.json"
            )
            projection["portable_source_sha256"] = _sha256(
                input_root / "runtime_teacher_model_provenance_authority.json"
            )
            projection["relocation"] = "bundle-relative-authority-v1"
            projection["model_path"] = str(model_root)
            projection["runtime_model_path_is_nonsemantic"] = True
            projection["workload_identity"] = authority["workload_identity"]
            projection["replay_identity"] = replay_identity
            projection["checkpoint_digest"] = checkpoint_digest
            projection["bundle_manifest_sha256"] = bundle_manifest_sha256
            projection_path = input_root / "teacher_model_provenance.json"
            projection_path.write_text(
                json.dumps(projection, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            provenance_report = validate_teacher_model_provenance(projection_path)
            if not provenance_report.ok:
                raise ValueError(
                    "adopted teacher provenance projection invalid: "
                    + "; ".join(provenance_report.blockers)
                )
            input_closure = {}
            for member in sorted(input_root.rglob("*")):
                if member.is_symlink():
                    raise ValueError("current replay adopted input contains symlink")
                if member.is_dir():
                    continue
                if not member.is_file():
                    raise ValueError(
                        "current replay adopted input contains special file"
                    )
                input_closure[member.relative_to(adopted_root).as_posix()] = _sha256(
                    member
                )
            metadata = {
                "schema_version": "radjax_tome_frozen_selection_replay_v2",
                "provenance": authority["provenance"],
                "workload_identity": authority["workload_identity"],
                "bundle_manifest_sha256": bundle_manifest_sha256,
                "checkpoint_digest": checkpoint_digest,
                "selected_record_digest": selected_record_digest,
                "replay_identity": replay_identity,
                "selected_sources": selected_sources,
                "selected_coordinates": selected_coordinates,
                "metadata_digest": metadata_digest,
                "input_root": "input",
                "input_closure": input_closure,
            }
            (adopted_root / "replay_authority.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
        return FrozenSelectionReplay(
            replay_root=replay_root,
            bundle_manifest=bundle_manifest,
            adopted_root=adopted_root,
            bundle_manifest_sha256=bundle_manifest_sha256,
            checkpoint_digest=checkpoint_digest,
            selected_record_digest=selected_record_digest,
            replay_identity=replay_identity,
            records=tuple(records),
            selected_sources=selected_sources,
            selected_coordinates=selected_coordinates,
            metadata_digest=metadata_digest,
        )
    layout = document.get("provenance", {}).get("artifact_layout", {})
    replay_relative = layout.get("replay_root")
    if not isinstance(replay_relative, str) or Path(replay_relative).is_absolute():
        raise ValueError("frozen replay layout is not relative")
    replay_root = (artifact_root / replay_relative).resolve()
    try:
        replay_root.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("frozen replay root escapes artifact root") from exc
    validator = _load_validator()
    validator.validate.artifact_root = artifact_root
    result = validator.validate(bundle_manifest)
    if result.get("status") != "HISTORICAL_M8_WORKLOAD_RECOVERED":
        raise ValueError("frozen replay must use recovered historical authority")
    if (
        result.get("selected_sources") != 213
        or result.get("selected_coordinates") != 256
    ):
        raise ValueError("frozen replay counts are not 213 sources and 256 coordinates")
    selected_artifact = load_multi_role_selection_artifact(
        replay_root / "c6" / "multi-role-selection"
    )
    records = tuple(
        dict(record)
        for record in c5_records_for_delivery(
            selected_artifact, delivery_path="two_pass_rerun_selected"
        )
    )
    selected_record_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    expected = document["normalized_inputs"]["selected_record_digest"]
    if selected_record_digest != expected:
        raise ValueError("frozen replay selected-record identity mismatch")
    bundle_manifest_sha256 = str(result["manifest_sha256"])
    checkpoint_digest = str(document["normalized_inputs"]["checkpoint_digest"])
    replay_identity = _replay_identity(
        bundle_manifest_sha256=bundle_manifest_sha256,
        checkpoint_digest=checkpoint_digest,
        selected_record_digest=selected_record_digest,
    )
    existing_metadata = adopted_root / "replay_authority.json"
    if adopted_root.exists():
        if adopted_root.is_symlink() or not adopted_root.is_dir():
            raise ValueError("private replay adoption root has the wrong type")
        if not existing_metadata.is_file() or existing_metadata.is_symlink():
            raise ValueError("private replay adoption is incomplete")
        prior = json.loads(existing_metadata.read_text(encoding="utf-8"))
        if (
            prior.get("bundle_manifest_sha256") != bundle_manifest_sha256
            or prior.get("checkpoint_digest") != checkpoint_digest
            or prior.get("selected_record_digest") != selected_record_digest
            or prior.get("replay_identity") != replay_identity
        ):
            raise ValueError("private replay adoption conflicts with frozen authority")
        for relative, expected_digest in (prior.get("member_digests") or {}).items():
            member = _owned_member_path(adopted_root, relative)
            if (
                member.is_symlink()
                or not member.is_file()
                or _sha256(member) != expected_digest
            ):
                raise ValueError(f"private replay adoption member mismatch: {relative}")
        return FrozenSelectionReplay(
            replay_root=replay_root,
            bundle_manifest=bundle_manifest,
            adopted_root=adopted_root,
            bundle_manifest_sha256=bundle_manifest_sha256,
            checkpoint_digest=checkpoint_digest,
            selected_record_digest=selected_record_digest,
            replay_identity=replay_identity,
            records=records,
            selected_sources=213,
            selected_coordinates=256,
        )
    adopted_root.mkdir(parents=True, exist_ok=False)
    try:
        for relative in (
            "c6/claims/selected_coordinates.jsonl",
            "c6/multi-role-selection/selected_exemplars.jsonl",
            "c6/source-passports.jsonl",
            "input/corpus.jsonl",
            "input/corpus_manifest.json",
            "input/teacher_model_provenance.json",
        ):
            source = replay_root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(
                    f"frozen replay member is not a regular file: {relative}"
                )
            target = adopted_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        metadata = {
            "schema_version": "radjax_tome_frozen_selection_replay_v1",
            "provenance": "HISTORICAL_M8_WORKLOAD_RECOVERED",
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "checkpoint_digest": checkpoint_digest,
            "selected_record_digest": selected_record_digest,
            "replay_identity": replay_identity,
            "selected_sources": 213,
            "selected_coordinates": 256,
            "member_digests": {
                relative: _sha256(adopted_root / relative)
                for relative in (
                    "c6/claims/selected_coordinates.jsonl",
                    "c6/multi-role-selection/selected_exemplars.jsonl",
                    "c6/source-passports.jsonl",
                    "input/corpus.jsonl",
                    "input/corpus_manifest.json",
                    "input/teacher_model_provenance.json",
                )
            },
        }
        (adopted_root / "replay_authority.json").write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(adopted_root, ignore_errors=True)
        raise
    return FrozenSelectionReplay(
        replay_root=replay_root,
        bundle_manifest=bundle_manifest,
        adopted_root=adopted_root,
        bundle_manifest_sha256=bundle_manifest_sha256,
        checkpoint_digest=checkpoint_digest,
        selected_record_digest=selected_record_digest,
        replay_identity=replay_identity,
        records=records,
        selected_sources=213,
        selected_coordinates=256,
    )
