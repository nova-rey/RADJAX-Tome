"""Closed dispatch for artifact forms promised by production contracts."""

from __future__ import annotations

import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def validate_artifact(
    path: Path,
    *,
    mode: str = "standard",
    expected: Path | None = None,
    attestation: Path | None = None,
) -> dict[str, Any]:
    """Validate v3/v4 production artifacts and canonical packages."""
    if mode not in {"standard", "governed", "external-attestation"}:
        raise ValueError("unsupported validation mode")
    if mode == "governed" and expected is None:
        raise ValueError("governed validation requires --expected")
    if mode == "external-attestation" and (attestation is None or expected is None):
        raise ValueError(
            "external-attestation validation requires --expected and --attestation"
        )
    candidate = _root(path.resolve())
    if candidate.is_file() and candidate.suffix == ".tgz":
        with tempfile.TemporaryDirectory(prefix="radjax-validate-") as directory:
            with tarfile.open(candidate, "r:*") as archive:
                archive.extractall(directory, filter="data")
            roots = list(Path(directory).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise ValueError("archive must contain one artifact root")
            return validate_artifact(
                roots[0], mode=mode, expected=expected, attestation=attestation
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
        from radjax_contract.tome.v3.validation import validate_tome_artifact_v3

        report = validate_tome_artifact_v3(candidate)
        return {
            "status": "pass" if report.ok else "fail",
            "kind": "contract_v3",
            "report": report.__dict__,
        }
    if candidate.is_file() and candidate.suffix == ".rtome":
        from radjax_contract.tome.v3.validation import validate_tome_artifact_v3

        try:
            report = validate_tome_artifact_v3(candidate)
            return {
                "status": "pass" if report.ok else "fail",
                "kind": "contract_v3",
                "report": report.__dict__,
            }
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
