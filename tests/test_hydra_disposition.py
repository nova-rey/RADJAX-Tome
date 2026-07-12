from __future__ import annotations

import json
import subprocess
from argparse import ArgumentParser, _SubParsersAction
from pathlib import Path

from radjax_tome.cli.main import _build_parser

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "hydra_disposition.json"
ALLOWED_STATUSES = {
    "canonical",
    "supporting",
    "research-frozen",
    "compatibility-only",
    "remove-after-parity",
}
REQUIRED_RECORD_KEYS = {
    "kind",
    "status",
    "rationale",
    "owning_milestone",
    "replacement_or_target_boundary",
    "archive_or_removal_condition",
}


def test_hydra_disposition_has_complete_tracked_inventory() -> None:
    records = _ledger()["records"]
    expected = _tracked_python_modules() | _tracked_scripts() | _top_level_docs()
    expected |= {f"cli:{command}" for command in _public_commands(_build_parser())}

    assert expected <= records.keys()


def test_hydra_disposition_schema_and_statuses_are_valid() -> None:
    payload = _ledger()

    assert payload["schema_version"] == "radjax_tome.hydra_disposition.v1"
    assert payload["baseline"] == "7a56a0808453f4b4ecc6cefe3ee63b724c701980"
    assert payload["known_dependency_boundary_violations"]

    for identifier, record in payload["records"].items():
        assert REQUIRED_RECORD_KEYS <= record.keys(), identifier
        assert record["kind"] in {
            "source-module",
            "script",
            "top-level-doc",
            "public-cli-command",
        }
        assert record["status"] in ALLOWED_STATUSES, identifier
        assert record["rationale"], identifier
        assert record["owning_milestone"], identifier


def test_initializer_boundary_violations_are_explicitly_owned() -> None:
    payload = _ledger()
    violations = {
        item["module"]: item["remediation_milestone"]
        for item in payload["known_dependency_boundary_violations"]
    }

    assert violations == {
        "src/radjax_tome/backends/__init__.py": "M6",
        "src/radjax_tome/builder/__init__.py": "M3/M6",
        "src/radjax_tome/reports/__init__.py": "M6",
    }
    for module in violations:
        assert payload["records"][module]["known_dependency_boundary_violation"]


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER.read_text(encoding="utf-8"))


def _git_files(*patterns: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *patterns],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return {line for line in result.stdout.splitlines() if line}


def _tracked_python_modules() -> set[str]:
    return {path for path in _git_files("src/radjax_tome") if path.endswith(".py")}


def _tracked_scripts() -> set[str]:
    return _git_files("scripts/*.py")


def _top_level_docs() -> set[str]:
    return {path for path in _git_files("docs") if len(Path(path).parts) == 2}


def _public_commands(parser: ArgumentParser, prefix: str = "") -> set[str]:
    commands: set[str] = set()
    for action in parser._actions:
        if not isinstance(action, _SubParsersAction):
            continue
        for name, child in action.choices.items():
            command = f"{prefix} {name}".strip()
            commands.add(command)
            commands |= _public_commands(child, command)
    return commands
