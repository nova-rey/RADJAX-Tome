"""Small, stable checks for the user-facing M12 documentation surface."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from argparse import _SubParsersAction
from pathlib import Path

from radjax_tome.builder.config_io import load_tome_build_intent
from radjax_tome.cli.mainline import parser as mainline_parser

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _root_commands() -> set[str]:
    parser = mainline_parser()
    action = next(
        item for item in parser._actions if isinstance(item, _SubParsersAction)
    )
    return set(action.choices)


def _local_target(document: Path, target: str) -> Path | None:
    target = target.strip().strip("<>").split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if target.startswith(("docs/", "src/", "tests/", "scripts/")):
        return ROOT / target
    return document.parent / target


def test_documented_local_links_and_required_entry_points_exist() -> None:
    required = (
        "START_HERE.md",
        "CONFIGURATION_REFERENCE.md",
        "ARTIFACTS_AND_PACKAGES.md",
        "RESEARCH_STATUS_MAP.md",
        "CLI_GUIDE.md",
        "CORPUS_BUILDER.md",
    )
    missing = [str(DOCS / name) for name in required if not (DOCS / name).is_file()]
    assert not missing, "missing M12 documentation entry points: " + ", ".join(missing)

    broken: list[str] = []
    for document in DOCS.rglob("*.md"):
        for raw_target in LOCAL_LINK.findall(document.read_text(encoding="utf-8")):
            target = _local_target(document, raw_target)
            if target is not None and not target.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "broken local documentation links: " + "; ".join(broken)


def test_complete_documented_config_examples_load_canonically() -> None:
    examples = sorted(
        path
        for path in (DOCS / "examples").rglob("*")
        if path.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    assert examples, "no canonical configuration examples were found"
    for path in examples:
        intent = load_tome_build_intent(path)
        assert intent.schema_version in {
            "radjax_tome_build_intent_v1",
            "radjax_tome_build_intent_v2",
        }, path


def test_documented_cli_grammar_and_hydra_inventory_are_consistent() -> None:
    commands = _root_commands()
    ledger = json.loads((DOCS / "hydra_disposition.json").read_text(encoding="utf-8"))
    records = ledger["records"]
    declared = set(ledger["m9_public_surface"]["commands"])
    assert declared <= commands
    for command in declared:
        assert f"cli:{command}" in records or (
            command == "package" and "cli:package-artifact" in records
        )

    guide = (DOCS / "CLI_GUIDE.md").read_text(encoding="utf-8")
    for command in commands:
        assert command in guide

    parser = mainline_parser()
    for argv in (
        ["build", "--config", "intent.json"],
        ["corpus", "build", "--config", "corpus.json"],
        ["validate", "artifact"],
        ["inspect", "artifact"],
        ["package", "workspace", "--output", "package.tgz", "--profile", "student"],
    ):
        assert parser.parse_args(argv).command in commands


def test_examples_have_stable_execution_classifications_and_optional_tui_isolated() -> (
    None
):
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [ROOT / "README.md", *DOCS.glob("*.md")]
    ).lower()
    for marker in ("executed", "canonical-loader-validated", "template"):
        assert marker in text, (
            f"documentation is missing example classification: {marker}"
        )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, radjax_tome; assert 'textual' not in sys.modules",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
