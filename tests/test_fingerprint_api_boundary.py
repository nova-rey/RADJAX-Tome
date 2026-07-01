from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import radjax_tome.fingerprint as fingerprint

ROOT = Path(__file__).resolve().parents[1]
SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT / "src"),
}
RECOMMENDED_NAMES = {
    "FingerprintValidationResult",
    "build_minimal_fingerprint_artifact_from_target_store",
    "generate_corridor_measurement_report",
    "generate_corridor_subset_receipt",
    "generate_exemplar_reservoir",
    "inspect_fingerprint_artifact",
    "summarize_exemplar_reservoir",
    "validate_fingerprint_artifact",
}
REQUIRED_DOC_SECTIONS = (
    "## Recommended Imports",
    "## Recommended Workflow",
    "## Advanced Module Imports",
    "## What Is Not Public API",
)


def test_fingerprint_all_is_explicit_and_small() -> None:
    assert isinstance(fingerprint.__all__, list)
    assert set(fingerprint.__all__) == RECOMMENDED_NAMES
    assert len(fingerprint.__all__) <= 20
    assert all(not name.startswith("_") for name in fingerprint.__all__)


def test_recommended_public_names_import_from_package_root() -> None:
    for name in RECOMMENDED_NAMES:
        assert getattr(fingerprint, name).__name__ == name


def test_advanced_names_import_from_explicit_submodules() -> None:
    advanced_imports = {
        "radjax_tome.fingerprint.artifacts": "FingerprintManifest",
        "radjax_tome.fingerprint.corridor": "CorridorMeasurementReport",
        "radjax_tome.fingerprint.exemplars": "FingerprintExemplarRecord",
        "radjax_tome.fingerprint.generation": "generate_exemplar_reservoir",
        "radjax_tome.fingerprint.loader": "LoadedFingerprintArtifact",
    }

    for module_name, symbol in advanced_imports.items():
        module = importlib.import_module(module_name)
        assert getattr(module, symbol).__name__ == symbol


def test_representative_advanced_name_is_not_exported_from_root() -> None:
    assert "FingerprintManifest" not in fingerprint.__all__
    assert not hasattr(fingerprint, "FingerprintManifest")


def test_importing_fingerprint_root_does_not_import_heavy_optional_deps() -> None:
    script = (
        "import sys; "
        "import radjax_tome.fingerprint; "
        "bad=[name for name in ('torch','transformers','jax') if name in sys.modules]; "
        "raise SystemExit(1 if bad else 0)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=SUBPROCESS_ENV,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_fingerprint_api_doc_contains_required_boundary_sections() -> None:
    doc = ROOT / "docs" / "FINGERPRINT_API.md"
    text = doc.read_text(encoding="utf-8")

    for section in REQUIRED_DOC_SECTIONS:
        assert section in text
    assert "Artifact schemas and generated behavior are unchanged" in text
    assert "Deeper modules remain available for advanced use" in text


def test_cli_guide_points_to_fingerprint_api_doc() -> None:
    text = (ROOT / "docs" / "CLI_GUIDE.md").read_text(encoding="utf-8")

    assert "docs/FINGERPRINT_API.md" in text
