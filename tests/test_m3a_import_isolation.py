"""M3A subprocess characterization for the M3B import-boundary change.

These tests deliberately describe the pre-isolation surface.  M3B will replace
the eager-research assertions with isolation assertions while retaining the
subprocess harness, parser inventory, and explicit compatibility imports.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_IMPORTS = (
    "radjax_tome.builder.production",
    "radjax_tome.builder.corridor_artifacts",
    "radjax_tome.backends.gpu_torch",
    "radjax_tome.reports.run_plan",
    "radjax_tome.audit.selected_linkage",
    "radjax_tome.tome.packaging",
)

# These are deliberately distinct from the canonical leaves above.  They are
# public today, but M3B must make their package-facade access lazy.
RESEARCH_OR_COMPATIBILITY_EDGES = {
    "radjax_tome.audit.refactor_surface",
    "radjax_tome.backends.hf_export",
    "radjax_tome.backends.hf_specimen",
    "radjax_tome.backends.qwen_policy",
    "radjax_tome.builder.multi_gpu_path_b",
    "radjax_tome.fingerprint.artifacts",
    "radjax_tome.fingerprint.exemplars",
    "radjax_tome.fingerprint.generation",
    "radjax_tome.fingerprint.inspection",
    "radjax_tome.reports.arc",
    "radjax_tome.reports.baseline",
    "radjax_tome.reports.fingerprint_quality",
}

COMPATIBILITY_IMPORTS = (
    ("radjax_tome.backends.hf_export", "HFTeacherExportConfig"),
    ("radjax_tome.backends.hf_specimen", "HFTeacherSpecimenConfig"),
    ("radjax_tome.backends.qwen_policy", "QwenPolicyMap"),
    ("radjax_tome.builder.multi_gpu_path_b", "MultiGPUPathBConfig"),
    ("radjax_tome.reports.arc", "FingerprintArcReport"),
    ("radjax_tome.reports.baseline", "FingerprintBaselineComparisonReport"),
    (
        "radjax_tome.reports.fingerprint_quality",
        "FingerprintQualityPerByteReport",
    ),
    ("radjax_tome.audit.refactor_surface", "RefactorAudit"),
    ("radjax_tome.fingerprint.artifacts", "FingerprintManifest"),
    ("radjax_tome.fingerprint.exemplars", "FingerprintExemplarRecord"),
    ("radjax_tome.fingerprint.generation", "generate_exemplar_reservoir"),
    ("radjax_tome.fingerprint.inspection", "inspect_fingerprint_artifact"),
)

EXPECTED_COMMANDS = (
    "allocate-fingerprint-corridor-coverage",
    "audit-selected-linkage",
    "build",
    "build-fingerprint-corridor-leaderboards",
    "build-multi-role-selected-exemplars",
    "claim-corridor-and-backfill-global",
    "corpus",
    "doctor",
    "exemplar-delivery-parity",
    "export-production-global-board-supply",
    "golden",
    "inspect",
    "model",
    "multi-gpu-path-b",
    "pack",
    "package-artifact",
    "parity",
    "plan",
    "production-build",
    "prove-capabilities",
    "unpack",
    "validate",
    "validate-package",
)

_PROBE = r"""
import contextlib
import importlib
import io
import json
import sys

request = json.loads(sys.argv[1])
case = request["case"]
commands = []
help_text = ""
requested_symbol_present = None

if case == "root":
    package = importlib.import_module("radjax_tome")
    requested_symbol_present = all(
        hasattr(package, name)
        for name in (
            "FakeTeacherBackend",
            "TeacherTextbookBuildConfig",
            "build_teacher_textbook",
            "emit_toy_teacher_tome",
        )
    )
elif case == "canonical":
    importlib.import_module(request["module"])
elif case == "parser":
    module = importlib.import_module("radjax_tome.cli.main")
    parser = module._build_parser()
    commands = sorted(
        choice
        for action in parser._actions
        for choice in (getattr(action, "choices", None) or {})
    )
elif case == "help":
    module = importlib.import_module("radjax_tome.cli.main")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            module.main(["--help"])
        except SystemExit as error:
            assert error.code == 0
    help_text = output.getvalue()
elif case == "compatibility":
    module = importlib.import_module(request["module"])
    requested_symbol_present = hasattr(module, request["symbol"])
else:
    raise AssertionError(f"unknown probe case: {case}")

loaded = set(sys.modules)
project_modules = sorted(name for name in loaded if name.startswith("radjax_tome"))
optional_ml_modules = sorted(
    name
    for name in loaded
    if name == "torch" or name.startswith(("torch.", "transformers"))
)
print(
    "M3A_IMPORT_SNAPSHOT="
    + json.dumps(
        {
            "commands": commands,
            "help_text": help_text,
            "optional_ml_modules": optional_ml_modules,
            "project_modules": project_modules,
            "requested_symbol_present": requested_symbol_present,
        },
        sort_keys=True,
    )
)
"""


def _snapshot(case: str, **request: str) -> Mapping[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps({"case": case, **request})],
        cwd=REPOSITORY_ROOT,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.splitlines()
    encoded = next(
        line.removeprefix("M3A_IMPORT_SNAPSHOT=")
        for line in reversed(lines)
        if line.startswith("M3A_IMPORT_SNAPSHOT=")
    )
    return json.loads(encoded)


def _project_modules(snapshot: Mapping[str, object]) -> set[str]:
    return set(snapshot["project_modules"])


def test_m3a_root_snapshot_preserves_public_surface_and_eager_edges() -> None:
    snapshot = _snapshot("root")

    assert snapshot["requested_symbol_present"] is True
    assert snapshot["optional_ml_modules"] == []
    assert RESEARCH_OR_COMPATIBILITY_EDGES <= _project_modules(snapshot)


@pytest.mark.parametrize("module", CANONICAL_IMPORTS)
def test_m3a_canonical_import_snapshots_are_subprocess_isolated(module: str) -> None:
    snapshot = _snapshot("canonical", module=module)

    assert module in _project_modules(snapshot)
    assert snapshot["optional_ml_modules"] == []
    # M3A characterization: the root facade causes every canonical import to
    # pull the research/compatibility set.  M3B replaces this equality with a
    # negative isolation assertion without changing the probe contract.
    assert RESEARCH_OR_COMPATIBILITY_EDGES <= _project_modules(snapshot)


def test_m3a_parser_and_help_keep_the_complete_command_inventory() -> None:
    parser_snapshot = _snapshot("parser")
    help_snapshot = _snapshot("help")

    assert tuple(parser_snapshot["commands"]) == EXPECTED_COMMANDS
    assert "Recommended commands:" in str(help_snapshot["help_text"])
    for command in EXPECTED_COMMANDS:
        assert command in str(help_snapshot["help_text"])
    assert parser_snapshot["optional_ml_modules"] == []
    assert help_snapshot["optional_ml_modules"] == []
    assert RESEARCH_OR_COMPATIBILITY_EDGES <= _project_modules(parser_snapshot)
    assert RESEARCH_OR_COMPATIBILITY_EDGES <= _project_modules(help_snapshot)


@pytest.mark.parametrize(("module", "symbol"), COMPATIBILITY_IMPORTS)
def test_m3a_explicit_research_and_compatibility_modules_remain_importable(
    module: str,
    symbol: str,
) -> None:
    snapshot = _snapshot("compatibility", module=module, symbol=symbol)

    assert module in _project_modules(snapshot)
    assert snapshot["requested_symbol_present"] is True
    assert snapshot["optional_ml_modules"] == []
