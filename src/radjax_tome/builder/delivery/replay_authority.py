"""Strict adoption of an already-verified frozen selected-source authority."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radjax_tome.builder.c6_integration import c5_records_for_delivery
from radjax_tome.corpora import validate_corpus_artifact
from radjax_tome.fingerprint.multi_role_selection import (
    load_multi_role_selection_artifact,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
        if authority.get("provenance") != "NEW_DETERMINISTIC_M8G_1K_WORKLOAD":
            raise ValueError("current replay workload provenance invalid")
        selected_sources = int(authority["counts"]["selected_sources"])
        selected_coordinates = int(authority["counts"]["selected_coordinates"])
        replay_root = artifact_root
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
        records = [
            json.loads(x) for x in record_file.read_text().splitlines() if x.strip()
        ]
        if len(coords) != selected_coordinates or len(records) != selected_sources:
            raise ValueError("current replay selection counts mismatch")
        selected_record_digest = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
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
            for relative, expected_digest in (prior.get("input_closure") or {}).items():
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
                shutil.copytree(replay_root / relative, input_root / relative)
            for relative in (
                "runtime_teacher_model_provenance_authority.json",
                "teacher_identity.json",
                "runtime_teacher_model_provenance.json",
                "portable_path_policy.json",
            ):
                target = input_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(replay_root / relative, target)
            for source_name, target_name in (
                ("corpus/corpus.jsonl", "corpus.jsonl"),
                ("corpus/corpus_manifest.json", "corpus_manifest.json"),
                ("corpus/corpus_build_report.json", "corpus_build_report.json"),
                (
                    "runtime_teacher_model_provenance.json",
                    "teacher_model_provenance.json",
                ),
            ):
                shutil.copy2(input_root / source_name, input_root / target_name)
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
                "input_root": "input",
                "input_closure": {
                    f"input/{relative}": _sha256(adopted_root / "input" / relative)
                    for relative in (
                        "corpus/corpus.jsonl",
                        "corpus/corpus_manifest.json",
                        "corpus/corpus_build_report.json",
                        "corpus.jsonl",
                        "corpus_manifest.json",
                        "corpus_build_report.json",
                        "teacher_model_provenance.json",
                        "teacher_identity.json",
                        "runtime_teacher_model_provenance_authority.json",
                    )
                },
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
