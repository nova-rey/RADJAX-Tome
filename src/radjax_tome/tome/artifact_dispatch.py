"""Closed dispatch for artifact forms promised by production contracts."""

from __future__ import annotations

import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from radjax_tome.tome.archive_compat import safe_extractall
from radjax_tome.tome.packaging import validate_tome_package


@dataclass(frozen=True)
class ArtifactInspection:
    kind: str
    path: Path
    validation: dict[str, Any]
    metadata: dict[str, Any]


def _root(path: Path) -> Path:
    if path.is_dir():
        return path
    if path.is_file() and path.suffix in {".tgz", ".rtome"}:
        return path
    raise ValueError("unsupported artifact path")


def _validate_v3_mode(
    candidate: Path,
    *,
    mode: str,
    expected: Path | None,
    attestation: Path | None,
    attestation_policy: str,
    evaluation_time: str | None,
) -> dict[str, Any]:
    """Run the pinned Contract v3 mode API without weakening its semantics."""
    from radjax_contract.tome.v3 import (
        AttestationRequirement,
        compare_governed_tome_artifact_v3,
        validate_tome_artifact_v3,
        verify_external_tome_attestation_v3,
    )

    standard = validate_tome_artifact_v3(candidate)
    if mode == "standard":
        return {
            "status": "pass" if getattr(standard, "ok", False) else "fail",
            "kind": "contract_v3",
            "report": standard.__dict__,
        }
    if mode == "governed":
        assert expected is not None
        governed = compare_governed_tome_artifact_v3(candidate, expected)
        return {
            "status": "pass" if getattr(governed, "ok", False) else "fail",
            "kind": "contract_v3",
            "report": governed.__dict__,
        }
    assert attestation is not None and evaluation_time is not None
    evaluated = datetime.fromisoformat(evaluation_time.replace("Z", "+00:00"))
    if evaluated.tzinfo is None:
        raise ValueError("evaluation time must include timezone")
    external = verify_external_tome_attestation_v3(
        candidate,
        attestation,
        requirement=AttestationRequirement(attestation_policy),
        evaluation_time_utc=evaluated,
    )
    return {
        "status": "pass" if getattr(external, "ok", False) else "fail",
        "kind": "contract_v3",
        "report": external.__dict__,
    }


def validate_artifact(
    path: Path,
    *,
    mode: str = "standard",
    expected: Path | None = None,
    attestation: Path | None = None,
    attestation_policy: str = "optional",
    evaluation_time: str | None = None,
) -> dict[str, Any]:
    """Validate v3/v4 production artifacts and canonical packages."""
    if mode not in {"standard", "governed", "external-attestation"}:
        raise ValueError("unsupported validation mode")
    if attestation_policy not in {"optional", "required"}:
        raise ValueError("unsupported attestation policy")
    if mode == "governed" and expected is None:
        raise ValueError("governed validation requires --expected")
    if mode == "external-attestation" and attestation is None:
        raise ValueError("external-attestation validation requires --attestation")
    if mode == "external-attestation" and evaluation_time is None:
        raise ValueError("external-attestation validation requires --evaluation-time")
    if attestation_policy == "required" and attestation is None:
        raise ValueError("required attestation policy needs --attestation")
    if evaluation_time is not None:
        try:
            datetime.fromisoformat(evaluation_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("evaluation time must be RFC3339") from exc
    candidate = _root(path.resolve())
    if candidate.is_file() and candidate.suffix == ".tgz":
        with tempfile.TemporaryDirectory(prefix="radjax-validate-") as directory:
            with tarfile.open(candidate, "r:*") as archive:
                safe_extractall(archive, directory)
            roots = list(Path(directory).iterdir())
            root = (
                roots[0] if len(roots) == 1 and roots[0].is_dir() else Path(directory)
            )
            return validate_artifact(
                root,
                mode=mode,
                expected=expected,
                attestation=attestation,
                attestation_policy=attestation_policy,
                evaluation_time=evaluation_time,
            )
    if candidate.is_dir() and (candidate / "cover_page.json").is_file():
        import json

        cover = json.loads((candidate / "cover_page.json").read_text())
        schema = cover.get("schema_version") if isinstance(cover, dict) else None
        package = cover.get("package") if isinstance(cover, dict) else None
        if schema == "radjax_tome_cover_v4":
            from radjax_contract.tome import validate_streaming_tome

            report = validate_streaming_tome(candidate)
            return {
                "status": "pass" if report.ok else "fail",
                "kind": "m7_v4",
                "report": report.__dict__,
            }
        if isinstance(package, dict) and package.get("profile") in {
            "student",
            "full_debug_provenance",
        }:
            report = validate_tome_package(candidate)
            return {
                "status": "pass" if report.ok else "fail",
                "kind": "package",
                "report": report.to_dict()
                if hasattr(report, "to_dict")
                else report.__dict__,
            }
        return _validate_v3_mode(
            candidate,
            mode=mode,
            expected=expected,
            attestation=attestation,
            attestation_policy=attestation_policy,
            evaluation_time=evaluation_time,
        )
    if candidate.is_dir() and (candidate / "corpus_cover.json").is_file():
        from radjax_tome.corpora.validation import validate_corpus_artifact_v2

        report = validate_corpus_artifact_v2(candidate)
        return {
            "status": "pass" if report.ok else "fail",
            "kind": "corpus_v2",
            "report": report.to_dict(),
        }
    if candidate.is_file() and candidate.suffix == ".rtome":
        try:
            return _validate_v3_mode(
                candidate,
                mode=mode,
                expected=expected,
                attestation=attestation,
                attestation_policy=attestation_policy,
                evaluation_time=evaluation_time,
            )
        except (OSError, TypeError, ValueError):
            pass
        from radjax_tome.tome.bundle import validate_tome_bundle

        report = validate_tome_bundle(candidate)
        return {
            "status": "pass" if report.ok else "fail",
            "kind": "contract_v3_bundle",
            "report": report.__dict__,
        }
    if candidate.is_dir() or candidate.suffix == ".tgz":
        from radjax_contract.tome import validate_streaming_tome

        report = validate_streaming_tome(candidate)
        return {
            "status": "pass"
            if getattr(report, "ok", False) or getattr(report, "status", None) == "pass"
            else "fail",
            "kind": "m7_v4",
            "report": report.__dict__,
        }
    raise ValueError("artifact form is not a promised production contract")


def inspect_artifact(path: Path) -> ArtifactInspection:
    result = validate_artifact(path)
    metadata: dict[str, Any] = {}
    candidate = path.resolve()
    if candidate.is_dir():
        import json

        for name in (
            "cover_page.json",
            "corpus_cover.json",
            "corpus_manifest.json",
            "metadata.json",
            "production_build_report.json",
        ):
            file = candidate / name
            if file.is_file():
                try:
                    metadata[name] = json.loads(file.read_text())
                except json.JSONDecodeError:
                    pass
    return ArtifactInspection(result["kind"], path, result, metadata)
