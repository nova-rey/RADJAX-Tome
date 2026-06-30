from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json"
MANIFEST_PATH = ROOT / "docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json"


def _is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def test_ledger_covers_every_quarantine_manifest_path() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    quarantine_paths = {
        item["new_path_or_quarantine_path"]
        for item in manifest
        if str(item.get("new_path_or_quarantine_path", "")).startswith("quarantine/")
    }
    entries = _ledger()["entries"]
    ledger_paths = {entry["old_path"] for entry in entries}

    assert quarantine_paths
    assert ledger_paths == quarantine_paths
    assert _ledger()["summary"]["total_quarantined_paths"] == len(quarantine_paths)


def test_ledger_entries_have_dispositions_and_tracked_outputs() -> None:
    allowed = {
        "promoted",
        "split_promoted",
        "kept_quarantined",
        "belongs_student",
        "belongs_contract",
        "deprecated",
        "deferred",
        "waived",
    }
    for entry in _ledger()["entries"]:
        assert entry["decision"] in allowed
        assert entry["reason"]
        assert Path(entry["old_path"]).is_file()
        assert entry["quarantine_retained"] is True
        if entry["decision"] in {"promoted", "split_promoted"}:
            assert entry["active_new_paths"]
        for path_value in entry["active_new_paths"]:
            path = ROOT / path_value
            assert path.exists(), path_value
            assert _is_git_tracked(path), path_value
        for path_value in entry["tests_added"]:
            path = ROOT / path_value
            assert path.exists(), path_value
            assert _is_git_tracked(path), path_value


def test_active_code_has_no_quarantine_or_heavy_default_imports() -> None:
    forbidden_project_imports = {"qrwkv_xla", "quarantine"}
    heavy_imports = {"jax", "torch", "transformers"}
    for root_name in ("src", "scripts"):
        for path in (ROOT / root_name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    child.parent = parent
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name.split(".")[0] for alias in node.names}
                    assert not forbidden_project_imports.intersection(names), path
                    if isinstance(getattr(node, "parent", None), ast.Module):
                        assert not heavy_imports.intersection(names), path
                elif isinstance(node, ast.ImportFrom) and node.module:
                    root_name = node.module.split(".")[0]
                    assert root_name not in forbidden_project_imports, path
                    if isinstance(getattr(node, "parent", None), ast.Module):
                        assert root_name not in heavy_imports, path


def test_quarantine_remains_non_importable_reference_material() -> None:
    quarantine_root = ROOT / "quarantine/qrwkv_xla"
    assert quarantine_root.is_dir()
    assert not any(path.suffix == ".py" for path in quarantine_root.rglob("*"))
    assert not (ROOT / "src/radjax_tome/quarantine").exists()
