"""C5 durable multi-role selected-exemplar records.

This module is an offline projection of a validated C4 claim result.  It does
not select coordinates, materialize teacher payloads, or change production
selected-exemplar output.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

from radjax_tome.fingerprint.corridor_claims import (
    CorridorGlobalClaimResult,
    SelectionObligation,
)
from radjax_tome.io.json import read_json_object, write_json

MULTI_ROLE_SELECTION_SCHEMA = "radjax.multi_role_selected_exemplar.v1"
MULTI_ROLE_SELECTION_VALIDATION_SCHEMA = (
    "radjax.multi_role_selected_exemplar_validation.v1"
)
LEGACY_PROJECTION_SCHEMA = "selected_exemplars_v1"
MANIFEST_FILENAME = "manifest.json"
RICH_RECORDS_FILENAME = "selected_exemplars.jsonl"
LEGACY_RECORDS_FILENAME = "legacy_selected_exemplars.json"
VALIDATION_FILENAME = "validation_report.json"
NOT_MATERIALIZED_STATUS = "not_materialized_in_c5"
CORRIDOR_ROLE = "fingerprint_corridor_representative"
GLOBAL_ROLE = "global_board"
_FORBIDDEN_KEYS = frozenset(
    {
        "logits",
        "dense_logits",
        "top_probs",
        "top_log_probs",
        "input_ids",
        "source_text",
        "payload_ref",
    }
)

SourcePassportIndex: TypeAlias = Mapping[tuple[str, int], Mapping[str, Any]]


class MultiRoleSelectionError(ValueError):
    """Actionable C5 record, provenance, or artifact error."""


@dataclass(frozen=True)
class MultiRoleSelectedExemplar:
    schema_version: str
    example_id: str
    position: int
    selection_index: int
    primary_claim: str
    selection_roles: tuple[str, ...]
    selection_obligations: tuple[SelectionObligation, ...]
    represented_fingerprint_corridor_ids: tuple[int, ...]
    global_board_ids: tuple[str, ...]
    source_passport: Mapping[str, Any]
    payload_identity: Mapping[str, Any]
    multi_role: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "position": self.position,
            "selection_index": self.selection_index,
            "primary_claim": self.primary_claim,
            "selection_roles": list(self.selection_roles),
            "selection_obligations": [
                obligation.to_dict() for obligation in self.selection_obligations
            ],
            "represented_fingerprint_corridor_ids": list(
                self.represented_fingerprint_corridor_ids
            ),
            "global_board_ids": list(self.global_board_ids),
            "source_passport": dict(self.source_passport),
            "payload_identity": dict(self.payload_identity),
            "multi_role": self.multi_role,
        }


@dataclass(frozen=True)
class MultiRoleSelectionArtifact:
    records: tuple[MultiRoleSelectedExemplar, ...]
    legacy_records: tuple[Mapping[str, Any], ...]
    source_provenance: Mapping[str, Any]
    summary: Mapping[str, Any]
    c4_claims_sha256: str
    production_grade: bool
    warnings: tuple[str, ...] = ()

    @property
    def schema_version(self) -> str:
        return MULTI_ROLE_SELECTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_provenance": dict(self.source_provenance),
            "summary": dict(self.summary),
            "c4_claims_sha256": self.c4_claims_sha256,
            "production_grade": self.production_grade,
            "warnings": list(self.warnings),
            "records": [record.to_dict() for record in self.records],
            "legacy_records": [dict(record) for record in self.legacy_records],
        }


@dataclass(frozen=True)
class MultiRoleSelectionValidationResult:
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MULTI_ROLE_SELECTION_VALIDATION_SCHEMA,
            "status": self.status,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summary": dict(self.summary),
        }


def build_multi_role_selected_exemplars(
    claims: CorridorGlobalClaimResult,
    *,
    source_passports: SourcePassportIndex | None = None,
    require_source_passports: bool | None = None,
) -> MultiRoleSelectionArtifact:
    """Convert C4 coordinates and obligations into one durable record each."""

    if not isinstance(claims, CorridorGlobalClaimResult):
        raise TypeError("claims must be CorridorGlobalClaimResult")
    require_passports = (
        source_passports is not None
        if require_source_passports is None
        else require_source_passports
    )
    records: list[MultiRoleSelectedExemplar] = []
    for coordinate in claims.selected_coordinates:
        try:
            obligations = _canonical_obligations(
                coordinate.obligations,
                primary_claim=coordinate.primary_claim,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise MultiRoleSelectionError(
                f"invalid C4 obligations for {coordinate.example_id}: {exc}"
            ) from exc
        corridor_ids, board_ids = _derived_source_lists(obligations)
        passport = _build_source_passport(
            coordinate.example_id,
            coordinate.position,
            corridor_ids,
            source_passports,
            require_passports=require_passports,
        )
        records.append(
            MultiRoleSelectedExemplar(
                schema_version=MULTI_ROLE_SELECTION_SCHEMA,
                example_id=coordinate.example_id,
                position=coordinate.position,
                selection_index=coordinate.claim_order,
                primary_claim=coordinate.primary_claim,
                selection_roles=_selection_roles(obligations),
                selection_obligations=obligations,
                represented_fingerprint_corridor_ids=corridor_ids,
                global_board_ids=board_ids,
                source_passport=passport,
                payload_identity={
                    "payload_key": _payload_key(
                        coordinate.example_id,
                        coordinate.position,
                    ),
                    "materialization_status": NOT_MATERIALIZED_STATUS,
                },
                multi_role=len(obligations) > 1,
            )
        )
    legacy = tuple(project_legacy_selected_exemplars_from_records(records))
    source_provenance = {
        "c4": dict(claims.source_provenance),
        "c4_claim_policy": claims.policy.to_dict(),
        "production_grade": claims.production_grade,
    }
    artifact = MultiRoleSelectionArtifact(
        records=tuple(records),
        legacy_records=legacy,
        source_provenance=source_provenance,
        summary=_summary_from_records(records, legacy),
        c4_claims_sha256=_claims_digest(claims),
        production_grade=claims.production_grade,
        warnings=(
            ()
            if claims.production_grade
            else ("C4 source is non-production and was explicitly retained",)
        ),
    )
    validation = validate_multi_role_selection_artifact(
        artifact,
        claims=claims,
        production_grade=claims.production_grade,
    )
    if validation.status == "fail":
        raise MultiRoleSelectionError(
            "cannot build invalid C5 records: " + "; ".join(validation.blockers)
        )
    return artifact


def project_legacy_selected_exemplars(
    artifact: MultiRoleSelectionArtifact,
) -> list[dict[str, Any]]:
    """Return the one-row-per-coordinate compatibility projection."""

    if not isinstance(artifact, MultiRoleSelectionArtifact):
        raise TypeError("artifact must be MultiRoleSelectionArtifact")
    return [dict(record) for record in artifact.legacy_records]


def project_legacy_selected_exemplars_from_records(
    records: tuple[MultiRoleSelectedExemplar, ...] | list[MultiRoleSelectedExemplar],
) -> list[dict[str, Any]]:
    """Build the flat projection without teacher payload fields."""

    projection: list[dict[str, Any]] = []
    for record in records:
        primary = (
            record.selection_obligations[0] if record.selection_obligations else None
        )
        row: dict[str, Any] = {
            "schema_version": LEGACY_PROJECTION_SCHEMA,
            "rank": record.selection_index + 1,
            "selection_index": record.selection_index,
            "selected_example_id": record.example_id,
            "selected_position": record.position,
            "selected_score": None if primary is None else primary.score,
            "primary_claim": record.primary_claim,
            "selection_roles": list(record.selection_roles),
            "selection_obligations": [
                obligation.to_dict() for obligation in record.selection_obligations
            ],
            "represented_fingerprint_corridor_ids": list(
                record.represented_fingerprint_corridor_ids
            ),
            "global_board_ids": list(record.global_board_ids),
            "source_passport": dict(record.source_passport),
            "payload_identity": dict(record.payload_identity),
            "multi_role": record.multi_role,
        }
        corridor_id = record.represented_fingerprint_corridor_ids
        if corridor_id:
            row["corridor_mode_id"] = corridor_id[0]
        for field_name in (
            "source_shard_id",
            "source_row",
            "source_position",
            "source_top_token_id",
            "source_score",
            "corridor_fingerprint_id",
            "corridor_assignment_status",
        ):
            if field_name in record.source_passport:
                row[field_name] = record.source_passport[field_name]
        projection.append(row)
    return projection


def validate_multi_role_selection_artifact(
    artifact_or_path: MultiRoleSelectionArtifact | str | Path,
    *,
    claims: CorridorGlobalClaimResult | None = None,
    production_grade: bool = True,
) -> MultiRoleSelectionValidationResult:
    """Validate an in-memory C5 artifact or its atomic on-disk form."""

    if isinstance(artifact_or_path, (str, Path)):
        return _validate_artifact_path(
            Path(artifact_or_path),
            claims=claims,
            production_grade=production_grade,
        )
    return _validate_artifact_object(
        artifact_or_path,
        claims=claims,
        production_grade=production_grade,
    )


def load_multi_role_selection_artifact(
    path: str | Path,
    *,
    production_grade: bool = True,
) -> MultiRoleSelectionArtifact:
    """Load a hash-validated C5 artifact."""

    validation = validate_multi_role_selection_artifact(
        path,
        production_grade=production_grade,
    )
    if validation.status == "fail":
        raise MultiRoleSelectionError(
            "cannot load invalid C5 artifact: " + "; ".join(validation.blockers)
        )
    return _read_artifact(Path(path))


def write_multi_role_selection_artifact(
    artifact: MultiRoleSelectionArtifact,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write rich records and their legacy flat projection."""

    if not isinstance(artifact, MultiRoleSelectionArtifact):
        raise TypeError("artifact must be MultiRoleSelectionArtifact")
    validation = validate_multi_role_selection_artifact(
        artifact,
        production_grade=artifact.production_grade,
    )
    if validation.status == "fail":
        raise MultiRoleSelectionError(
            "cannot write invalid C5 artifact: " + "; ".join(validation.blockers)
        )
    output = Path(output_dir)
    if output.exists() and not overwrite:
        raise ValueError(f"C5 output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rich_path = temporary / RICH_RECORDS_FILENAME
        rich_path.write_text(
            "".join(
                json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
                + "\n"
                for record in artifact.records
            ),
            encoding="utf-8",
        )
        legacy_path = temporary / LEGACY_RECORDS_FILENAME
        write_json(
            legacy_path,
            {
                "schema_version": LEGACY_PROJECTION_SCHEMA,
                "projection_policy": "one_coordinate_one_row_v1",
                "selected_exemplars": [dict(row) for row in artifact.legacy_records],
            },
        )
        files = {
            RICH_RECORDS_FILENAME: {"sha256": _sha256(rich_path)},
            LEGACY_RECORDS_FILENAME: {"sha256": _sha256(legacy_path)},
        }
        manifest = {
            "schema_version": MULTI_ROLE_SELECTION_SCHEMA,
            "source_provenance": dict(artifact.source_provenance),
            "summary": dict(artifact.summary),
            "c4_claims_sha256": artifact.c4_claims_sha256,
            "production_grade": artifact.production_grade,
            "warnings": list(artifact.warnings),
            "files": files,
        }
        write_json(temporary / MANIFEST_FILENAME, manifest)
        report = validation.to_dict()
        report["file_hashes"] = {
            name: details["sha256"] for name, details in files.items()
        }
        write_json(temporary / VALIDATION_FILENAME, report)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def inspect_multi_role_selection_artifact(path: str | Path) -> dict[str, Any]:
    """Return a compact C5 artifact summary."""

    root = Path(path)
    validation = validate_multi_role_selection_artifact(root, production_grade=False)
    manifest = read_json_object(root / MANIFEST_FILENAME)
    return {
        "status": validation.status,
        "blockers": list(validation.blockers),
        "warnings": list(validation.warnings),
        "production_grade": manifest.get("production_grade"),
        "summary": dict(manifest.get("summary") or {}),
        "c4_claims_sha256": manifest.get("c4_claims_sha256"),
    }


def payload_key_for_coordinate(example_id: str, position: int) -> str:
    """Return the stable C5 payload identity for one canonical coordinate."""

    return _payload_key(example_id, position)


def load_source_passport_index(
    path: str | Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load a JSON passport index for strict production linkage checks."""

    payload = read_json_object(Path(path))
    rows = payload.get("passports", payload)
    if isinstance(rows, Mapping):
        rows = list(rows.values())
    if not isinstance(rows, list):
        raise MultiRoleSelectionError("source passport index must contain passports")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise MultiRoleSelectionError("source passport row must be an object")
        try:
            key = (str(row["example_id"]), int(row["position"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MultiRoleSelectionError(
                "source passport requires example_id and position"
            ) from exc
        if key in result:
            raise MultiRoleSelectionError(
                f"duplicate source passport: {key[0]}:{key[1]}"
            )
        result[key] = dict(row)
    return result


def _validate_artifact_object(
    artifact: MultiRoleSelectionArtifact,
    *,
    claims: CorridorGlobalClaimResult | None,
    production_grade: bool,
) -> MultiRoleSelectionValidationResult:
    blockers: list[str] = []
    warnings = list(artifact.warnings)
    if not isinstance(artifact, MultiRoleSelectionArtifact):
        return MultiRoleSelectionValidationResult(
            status="fail", blockers=("artifact must be MultiRoleSelectionArtifact",)
        )
    if not isinstance(artifact.production_grade, bool):
        blockers.append("production_grade must be boolean")
    if artifact.production_grade != bool(
        artifact.source_provenance.get("production_grade")
    ):
        blockers.append("production_grade does not match source provenance")
    if not artifact.production_grade:
        if production_grade:
            blockers.append("C5 artifact is non-production")
        else:
            warnings.append("C5 artifact uses explicitly allowed non-production input")
    coordinates: list[tuple[str, int]] = []
    payload_keys: list[str] = []
    for expected_index, record in enumerate(artifact.records):
        if record.schema_version != MULTI_ROLE_SELECTION_SCHEMA:
            blockers.append(f"record {expected_index} has unsupported schema")
        if not record.example_id:
            blockers.append(f"record {expected_index} has empty example_id")
        if record.position < 0:
            blockers.append(f"record {expected_index} has negative position")
        if record.selection_index != expected_index:
            blockers.append(f"record {expected_index} selection order mismatch")
        coordinate = (record.example_id, record.position)
        coordinates.append(coordinate)
        try:
            ordered = _canonical_obligations(
                record.selection_obligations,
                primary_claim=record.primary_claim,
            )
            if ordered != record.selection_obligations:
                blockers.append(
                    f"record {expected_index} obligations are not canonically ordered"
                )
            roles = _selection_roles(ordered)
            corridor_ids, board_ids = _derived_source_lists(ordered)
            if tuple(record.selection_roles) != roles:
                blockers.append(f"record {expected_index} selection_roles mismatch")
            if tuple(record.represented_fingerprint_corridor_ids) != corridor_ids:
                blockers.append(
                    f"record {expected_index} corridor ID derivation mismatch"
                )
            if tuple(record.global_board_ids) != board_ids:
                blockers.append(f"record {expected_index} board ID derivation mismatch")
            if record.multi_role != (len(ordered) > 1):
                blockers.append(f"record {expected_index} multi_role mismatch")
        except (TypeError, ValueError, KeyError) as exc:
            blockers.append(f"record {expected_index} obligation error: {exc}")
        passport = dict(record.source_passport)
        if passport.get("example_id") != record.example_id:
            blockers.append(f"record {expected_index} source passport example mismatch")
        if passport.get("position") != record.position:
            blockers.append(
                f"record {expected_index} source passport position mismatch"
            )
        if (
            "source_position" in passport
            and passport["source_position"] != record.position
        ):
            blockers.append(f"record {expected_index} source position mismatch")
        if record.represented_fingerprint_corridor_ids and (
            passport.get("corridor_mode_id")
            != record.represented_fingerprint_corridor_ids[0]
        ):
            blockers.append(f"record {expected_index} corridor passport mismatch")
        identity = dict(record.payload_identity)
        payload_key = identity.get("payload_key")
        if not isinstance(payload_key, str) or not payload_key:
            blockers.append(f"record {expected_index} payload key missing")
        else:
            payload_keys.append(payload_key)
            if payload_key != _payload_key(record.example_id, record.position):
                blockers.append(f"record {expected_index} payload key mismatch")
        if identity.get("materialization_status") not in {
            NOT_MATERIALIZED_STATUS,
            "existing_payload_link_verified",
            "pending_path_b_rerun",
        }:
            blockers.append(f"record {expected_index} payload status unsupported")
        _reject_forbidden_fields(record.to_dict(), blockers, f"record {expected_index}")
    if len(coordinates) != len(set(coordinates)):
        blockers.append("duplicate canonical coordinates")
    if len(payload_keys) != len(set(payload_keys)):
        blockers.append("distinct coordinates share a payload key")
    if len(artifact.legacy_records) != len(artifact.records):
        blockers.append("legacy projection count does not match rich records")
    else:
        for index, (record, legacy) in enumerate(
            zip(artifact.records, artifact.legacy_records, strict=True)
        ):
            if (
                legacy.get("selected_example_id") != record.example_id
                or legacy.get("selected_position") != record.position
                or legacy.get("selection_index") != record.selection_index
                or legacy.get("primary_claim") != record.primary_claim
            ):
                blockers.append(f"legacy projection mismatch at record {index}")
            _reject_forbidden_fields(
                legacy,
                blockers,
                f"legacy record {index}",
            )
    expected_summary = _summary_from_records(artifact.records, artifact.legacy_records)
    if dict(artifact.summary) != expected_summary:
        blockers.append("C5 summary arithmetic is inconsistent")
    if claims is not None:
        _validate_against_claims(artifact, claims, blockers)
    status = "fail" if blockers else ("warn" if warnings else "pass")
    return MultiRoleSelectionValidationResult(
        status=status,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        summary=expected_summary,
    )


def _validate_against_claims(
    artifact: MultiRoleSelectionArtifact,
    claims: CorridorGlobalClaimResult,
    blockers: list[str],
) -> None:
    if artifact.c4_claims_sha256 != _claims_digest(claims):
        blockers.append("C4 claim result hash mismatch")
    if dict(artifact.source_provenance.get("c4") or {}) != dict(
        claims.source_provenance
    ):
        blockers.append("C4 source provenance mismatch")
    if artifact.production_grade != claims.production_grade:
        blockers.append("C5 production grade does not match C4")
    if not claims.selected_coordinates and artifact.records:
        blockers.append("C5 has records but C4 selected coordinates are empty")
    if len(artifact.records) != len(claims.selected_coordinates):
        blockers.append("C5 record count does not match C4 selected count")
        return
    for record, coordinate in zip(
        artifact.records,
        claims.selected_coordinates,
        strict=True,
    ):
        if (record.example_id, record.position) != (
            coordinate.example_id,
            coordinate.position,
        ):
            blockers.append(
                "C5 coordinate does not match C4 at selection index "
                f"{record.selection_index}"
            )
        if record.selection_index != coordinate.claim_order:
            blockers.append(
                f"C5 selection index does not match C4 at {record.example_id}"
            )
        if record.primary_claim != coordinate.primary_claim:
            blockers.append(
                f"C5 primary claim does not match C4 at {record.example_id}"
            )
        if tuple(
            obligation.to_dict() for obligation in record.selection_obligations
        ) != tuple(obligation.to_dict() for obligation in coordinate.obligations):
            blockers.append(f"C5 obligations do not match C4 at {record.example_id}")


def _validate_artifact_path(
    root: Path,
    *,
    claims: CorridorGlobalClaimResult | None,
    production_grade: bool,
) -> MultiRoleSelectionValidationResult:
    try:
        manifest = read_json_object(root / MANIFEST_FILENAME)
        if manifest.get("schema_version") != MULTI_ROLE_SELECTION_SCHEMA:
            raise MultiRoleSelectionError("unsupported C5 artifact schema")
        artifact = _read_artifact(root)
        blockers: list[str] = []
        for filename, info in (manifest.get("files") or {}).items():
            file_path = root / filename
            if not file_path.is_file() or info.get("sha256") != _sha256(file_path):
                blockers.append(f"C5 file hash mismatch: {filename}")
        report = read_json_object(root / VALIDATION_FILENAME)
        if report.get("schema_version") != MULTI_ROLE_SELECTION_VALIDATION_SCHEMA:
            blockers.append("unsupported C5 validation schema")
        validation = _validate_artifact_object(
            artifact,
            claims=claims,
            production_grade=production_grade,
        )
        expected_file_hashes = {
            name: details.get("sha256")
            for name, details in (manifest.get("files") or {}).items()
        }
        if report.get("file_hashes") != expected_file_hashes:
            blockers.append("C5 validation report file hashes mismatch")
        if report.get("summary") != dict(validation.summary):
            blockers.append("C5 validation report summary mismatch")
        blockers.extend(validation.blockers)
        status = "fail" if blockers else validation.status
        return MultiRoleSelectionValidationResult(
            status=status,
            blockers=tuple(dict.fromkeys(blockers)),
            warnings=validation.warnings,
            summary=validation.summary,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return MultiRoleSelectionValidationResult(
            status="fail", blockers=(f"C5 artifact unreadable: {exc}",)
        )


def _read_artifact(root: Path) -> MultiRoleSelectionArtifact:
    manifest = read_json_object(root / MANIFEST_FILENAME)
    records: list[MultiRoleSelectedExemplar] = []
    rich_path = root / RICH_RECORDS_FILENAME
    if not rich_path.is_file():
        raise MultiRoleSelectionError("selected_exemplars.jsonl is missing")
    for line_number, line in enumerate(
        rich_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(_record_from_dict(json.loads(line)))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise MultiRoleSelectionError(
                f"invalid C5 record line {line_number}: {exc}"
            ) from exc
    legacy_payload = read_json_object(root / LEGACY_RECORDS_FILENAME)
    legacy = legacy_payload.get("selected_exemplars")
    if not isinstance(legacy, list) or any(
        not isinstance(item, dict) for item in legacy
    ):
        raise MultiRoleSelectionError("legacy selected-exemplar projection is invalid")
    return MultiRoleSelectionArtifact(
        records=tuple(records),
        legacy_records=tuple(legacy),
        source_provenance=dict(manifest.get("source_provenance") or {}),
        summary=dict(manifest.get("summary") or {}),
        c4_claims_sha256=str(manifest.get("c4_claims_sha256") or ""),
        production_grade=bool(manifest.get("production_grade")),
        warnings=tuple(str(item) for item in manifest.get("warnings", [])),
    )


def _record_from_dict(payload: Mapping[str, Any]) -> MultiRoleSelectedExemplar:
    obligations = tuple(
        SelectionObligation(
            role=str(item["role"]),
            source_id=str(item["source_id"]),
            rank=int(item["rank"]),
            score=None if item.get("score") is None else float(item["score"]),
            metadata=dict(item.get("metadata") or {}),
        )
        for item in payload["selection_obligations"]
    )
    return MultiRoleSelectedExemplar(
        schema_version=str(payload["schema_version"]),
        example_id=str(payload["example_id"]),
        position=int(payload["position"]),
        selection_index=int(payload["selection_index"]),
        primary_claim=str(payload["primary_claim"]),
        selection_roles=tuple(str(item) for item in payload["selection_roles"]),
        selection_obligations=obligations,
        represented_fingerprint_corridor_ids=tuple(
            int(item) for item in payload["represented_fingerprint_corridor_ids"]
        ),
        global_board_ids=tuple(str(item) for item in payload["global_board_ids"]),
        source_passport=dict(payload["source_passport"]),
        payload_identity=dict(payload["payload_identity"]),
        multi_role=bool(payload["multi_role"]),
    )


def _canonical_obligations(
    obligations: tuple[SelectionObligation, ...],
    *,
    primary_claim: str,
) -> tuple[SelectionObligation, ...]:
    if primary_claim not in {CORRIDOR_ROLE, GLOBAL_ROLE}:
        raise ValueError(f"unsupported primary claim: {primary_claim}")
    corridor = [item for item in obligations if item.role == CORRIDOR_ROLE]
    global_items = [item for item in obligations if item.role == GLOBAL_ROLE]
    if len(corridor) + len(global_items) != len(obligations):
        raise ValueError("selection obligations contain an unsupported role")
    if len(corridor) > 1:
        raise ValueError("multiple corridor obligations are unsupported in C5")
    if primary_claim == CORRIDOR_ROLE and not corridor:
        raise ValueError("corridor primary claim has no corridor obligation")
    if primary_claim == GLOBAL_ROLE and corridor:
        raise ValueError("global primary claim cannot include a corridor obligation")
    if primary_claim == CORRIDOR_ROLE:
        ordered = sorted(corridor, key=_corridor_obligation_key)
        ordered.extend(sorted(global_items, key=_global_obligation_key))
    else:
        ordered = sorted(global_items, key=_global_obligation_key)
    if not ordered:
        raise ValueError("selected coordinate has no obligations")
    return tuple(ordered)


def _corridor_obligation_key(obligation: SelectionObligation) -> tuple[int, int, str]:
    mode = obligation.metadata.get("corridor_mode_id")
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError("corridor obligation is missing integer corridor_mode_id")
    return (mode, obligation.rank, obligation.source_id)


def _global_obligation_key(
    obligation: SelectionObligation,
) -> tuple[int, str, int]:
    priority = obligation.metadata.get("board_priority")
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ValueError("global obligation is missing integer board_priority")
    return (priority, obligation.source_id, obligation.rank)


def _selection_roles(obligations: tuple[SelectionObligation, ...]) -> tuple[str, ...]:
    roles: list[str] = []
    for role in (CORRIDOR_ROLE, GLOBAL_ROLE):
        if any(obligation.role == role for obligation in obligations):
            roles.append(role)
    return tuple(roles)


def _derived_source_lists(
    obligations: tuple[SelectionObligation, ...],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    corridor_ids: list[int] = []
    board_ids: list[str] = []
    for obligation in obligations:
        if obligation.role == CORRIDOR_ROLE:
            mode = obligation.metadata.get("corridor_mode_id")
            if isinstance(mode, bool) or not isinstance(mode, int):
                raise ValueError("corridor obligation is missing corridor_mode_id")
            if mode not in corridor_ids:
                corridor_ids.append(mode)
        elif obligation.source_id not in board_ids:
            board_ids.append(obligation.source_id)
    return tuple(sorted(corridor_ids)), tuple(board_ids)


def _build_source_passport(
    example_id: str,
    position: int,
    corridor_ids: tuple[int, ...],
    source_passports: SourcePassportIndex | None,
    *,
    require_passports: bool,
) -> dict[str, Any]:
    passport: dict[str, Any]
    if source_passports is None:
        passport = {}
    else:
        raw = source_passports.get((example_id, position))
        if raw is None:
            if require_passports:
                raise MultiRoleSelectionError(
                    f"missing source passport for {example_id}:{position}"
                )
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError("source passport must be a mapping")
        passport = dict(raw)
    passport_blockers: list[str] = []
    _reject_forbidden_fields(passport, passport_blockers, "source passport")
    if passport_blockers:
        raise MultiRoleSelectionError("; ".join(passport_blockers))
    if source_passports is not None and require_passports:
        if (
            passport.get("example_id") != example_id
            or passport.get("position") != position
        ):
            raise MultiRoleSelectionError(
                f"source passport identity mismatch for {example_id}:{position}"
            )
    if "example_id" in passport and passport["example_id"] != example_id:
        raise MultiRoleSelectionError(
            f"source passport example mismatch for {example_id}"
        )
    if "position" in passport and passport["position"] != position:
        raise MultiRoleSelectionError(
            f"source passport position mismatch for {example_id}"
        )
    passport["example_id"] = example_id
    passport["position"] = position
    if corridor_ids:
        mode = corridor_ids[0]
        if passport.get("corridor_mode_id", mode) != mode:
            raise MultiRoleSelectionError(
                f"source passport corridor mode mismatch for {example_id}:{position}"
            )
        if passport.get("corridor_assignment_status", "linked") != "linked":
            raise MultiRoleSelectionError(
                f"corridor assignment is not linked for {example_id}:{position}"
            )
        passport["corridor_mode_id"] = mode
        passport["corridor_assignment_status"] = "linked"
    else:
        passport.setdefault("corridor_mode_id", None)
        passport.setdefault("corridor_assignment_status", "not_applicable")
    return passport


def _summary_from_records(
    records: tuple[MultiRoleSelectedExemplar, ...] | list[MultiRoleSelectedExemplar],
    legacy_records: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
) -> dict[str, Any]:
    corridor_mode_counts: Counter[str] = Counter()
    global_board_counts: Counter[str] = Counter()
    payload_status_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    corridor_role_count = 0
    global_role_count = 0
    multi_role_count = 0
    obligation_count = 0
    for record in records:
        primary_counts[record.primary_claim] += 1
        corridor_role_count += CORRIDOR_ROLE in record.selection_roles
        global_role_count += GLOBAL_ROLE in record.selection_roles
        multi_role_count += record.multi_role
        obligation_count += len(record.selection_obligations)
        for mode in record.represented_fingerprint_corridor_ids:
            corridor_mode_counts[str(mode)] += 1
        for board in record.global_board_ids:
            global_board_counts[board] += 1
        payload_status_counts[
            str(record.payload_identity.get("materialization_status"))
        ] += 1
    return {
        "unique_selected_count": len(records),
        "legacy_projection_count": len(legacy_records),
        "primary_corridor_count": primary_counts[CORRIDOR_ROLE],
        "primary_global_count": primary_counts[GLOBAL_ROLE],
        "corridor_role_count": corridor_role_count,
        "global_role_count": global_role_count,
        "multi_role_coordinate_count": multi_role_count,
        "obligation_count": obligation_count,
        "counts_by_corridor_mode": dict(sorted(corridor_mode_counts.items())),
        "counts_by_global_board": dict(sorted(global_board_counts.items())),
        "payload_status_counts": dict(sorted(payload_status_counts.items())),
    }


def _payload_key(example_id: str, position: int) -> str:
    encoded = json.dumps(
        [example_id, position],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "coordinate_v1:" + hashlib.sha256(encoded).hexdigest()


def _claims_digest(claims: CorridorGlobalClaimResult) -> str:
    payload = json.dumps(
        claims.to_dict(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_forbidden_fields(
    value: Any,
    blockers: list[str],
    path: str,
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _FORBIDDEN_KEYS:
                blockers.append(f"forbidden field {path}.{key}")
            _reject_forbidden_fields(child, blockers, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, blockers, f"{path}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
