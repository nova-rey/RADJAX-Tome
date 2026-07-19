"""Fresh-process M3A characterization and staged M3B import-boundary gates."""

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

# The first M3B initializer slice isolates the root, backends, and audit
# facades. Builder, reports, and fingerprint isolation intentionally land in
# the second slice, so their final negative assertions belong in a later gate.
FIRST_SLICE_CANONICAL_IMPORTS = (
    "radjax_tome.backends.gpu_torch",
    "radjax_tome.audit.selected_linkage",
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

FIRST_SLICE_COMPATIBILITY_IMPORTS = (
    ("radjax_tome.backends.hf_export", "HFTeacherExportConfig"),
    ("radjax_tome.backends.hf_specimen", "HFTeacherSpecimenConfig"),
    ("radjax_tome.backends.qwen_policy", "QwenPolicyMap"),
    ("radjax_tome.audit.refactor_surface", "RefactorAudit"),
)

FIRST_SLICE_PACKAGE_EXPORTS = (
    ("radjax_tome", "FakeTeacherBackend", "radjax_tome.backends.fake"),
    (
        "radjax_tome",
        "TeacherTextbookBuildConfig",
        "radjax_tome.builder.teacher_textbook",
    ),
    (
        "radjax_tome",
        "build_teacher_textbook",
        "radjax_tome.builder.teacher_textbook",
    ),
    (
        "radjax_tome",
        "emit_toy_teacher_tome",
        "radjax_tome.emit.teacher_tome",
    ),
    (
        "radjax_tome.backends",
        "HFTeacherExportConfig",
        "radjax_tome.backends.hf_export",
    ),
    (
        "radjax_tome.backends",
        "HFTeacherSpecimenConfig",
        "radjax_tome.backends.hf_specimen",
    ),
    (
        "radjax_tome.backends",
        "QwenPolicyMap",
        "radjax_tome.backends.qwen_policy",
    ),
    (
        "radjax_tome.audit",
        "RefactorAudit",
        "radjax_tome.audit.refactor_surface",
    ),
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
project_modules_before_symbol = []

if case == "root":
    importlib.import_module("radjax_tome")
elif case == "canonical":
    importlib.import_module(request["module"])
elif case == "gpu_availability_dispatch":
    module = importlib.import_module("radjax_tome.backends.gpu_torch")
    config = module.TeacherBackendConfig(
        backend_id="gpu_torch",
        runtime_mode="cpu_gpu",
        target_policy="corridor_exemplar_v1",
        model_id="missing-local-model",
        tokenizer_id="missing-local-tokenizer",
        local_files_only=True,
        allow_downloads=False,
    )
    module.check_gpu_torch_backend_available(config)
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
elif case == "package_compatibility":
    module = importlib.import_module(request["module"])
    project_modules_before_symbol = sorted(
        name for name in sys.modules if name.startswith("radjax_tome")
    )
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
            "project_modules_before_symbol": project_modules_before_symbol,
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


def _assert_isolated(snapshot: Mapping[str, object]) -> None:
    assert snapshot["optional_ml_modules"] == []
    assert not RESEARCH_OR_COMPATIBILITY_EDGES & _project_modules(snapshot)


def test_m3b_first_slice_root_snapshot_isolates_all_classified_leaves() -> None:
    snapshot = _snapshot("root")

    assert "radjax_tome" in _project_modules(snapshot)
    _assert_isolated(snapshot)


@pytest.mark.parametrize("module", CANONICAL_IMPORTS)
def test_m3a_canonical_import_snapshots_are_subprocess_isolated(module: str) -> None:
    snapshot = _snapshot("canonical", module=module)

    assert module in _project_modules(snapshot)
    assert snapshot["optional_ml_modules"] == []


@pytest.mark.parametrize("module", FIRST_SLICE_CANONICAL_IMPORTS)
def test_m3b_first_slice_canonical_imports_are_isolated(module: str) -> None:
    snapshot = _snapshot("canonical", module=module)

    assert module in _project_modules(snapshot)
    _assert_isolated(snapshot)


def test_m3b_first_slice_parser_and_help_keep_inventory_and_isolation() -> None:
    parser_snapshot = _snapshot("parser")
    help_snapshot = _snapshot("help")

    assert tuple(parser_snapshot["commands"]) == EXPECTED_COMMANDS
    assert "Recommended commands:" in str(help_snapshot["help_text"])
    for command in EXPECTED_COMMANDS:
        assert command in str(help_snapshot["help_text"])
    assert len(EXPECTED_COMMANDS) == 23
    _assert_isolated(parser_snapshot)
    _assert_isolated(help_snapshot)


def test_m3b_first_slice_native_gpu_availability_dispatch_has_no_research_leakage() -> (
    None
):
    snapshot = _snapshot("gpu_availability_dispatch")

    assert "radjax_tome.backends.gpu_torch" in _project_modules(snapshot)
    assert not RESEARCH_OR_COMPATIBILITY_EDGES & _project_modules(snapshot)


@pytest.mark.parametrize(("module", "symbol"), COMPATIBILITY_IMPORTS)
def test_m3a_explicit_research_and_compatibility_modules_remain_importable(
    module: str,
    symbol: str,
) -> None:
    snapshot = _snapshot("compatibility", module=module, symbol=symbol)

    assert module in _project_modules(snapshot)
    assert snapshot["requested_symbol_present"] is True
    assert snapshot["optional_ml_modules"] == []


@pytest.mark.parametrize(("module", "symbol"), FIRST_SLICE_COMPATIBILITY_IMPORTS)
def test_m3b_first_slice_direct_compatibility_leaves_are_explicit_and_isolated(
    module: str,
    symbol: str,
) -> None:
    snapshot = _snapshot("compatibility", module=module, symbol=symbol)

    assert snapshot["requested_symbol_present"] is True
    assert module in _project_modules(snapshot)
    assert not (RESEARCH_OR_COMPATIBILITY_EDGES - {module}) & _project_modules(snapshot)
    assert snapshot["optional_ml_modules"] == []


@pytest.mark.parametrize(
    ("module", "symbol", "leaf_module"), FIRST_SLICE_PACKAGE_EXPORTS
)
def test_m3b_first_slice_package_compatibility_exports_are_lazy(
    module: str,
    symbol: str,
    leaf_module: str,
) -> None:
    snapshot = _snapshot("package_compatibility", module=module, symbol=symbol)
    before_symbol = set(snapshot["project_modules_before_symbol"])

    assert snapshot["requested_symbol_present"] is True
    assert leaf_module not in before_symbol
    assert leaf_module in _project_modules(snapshot)
    assert snapshot["optional_ml_modules"] == []
