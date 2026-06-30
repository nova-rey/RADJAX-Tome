from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import radjax_tome

ROOT = Path(__file__).resolve().parents[1]


def _is_git_tracked(path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_normal_import_does_not_import_quarantine_modules() -> None:
    assert radjax_tome.__name__ == "radjax_tome"
    assert not any(name.startswith("radjax_tome.quarantine") for name in sys.modules)
    assert not any(name.startswith("qrwkv_xla") for name in sys.modules)


def test_bulk_manifest_covers_quarantine_reasons() -> None:
    manifest_path = Path("docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest
    allowed = {
        "migrate_now",
        "quarantine_for_surgery",
        "belongs_contract",
        "belongs_student",
        "defer_with_reason",
        "waive_with_reason",
        "already_migrated",
    }
    for item in manifest:
        assert item["classification"] in allowed
        assert item["old_path"]
        assert item["reason"]
        if item["classification"] == "quarantine_for_surgery":
            path = Path(item["new_path_or_quarantine_path"])
            assert str(path).startswith("quarantine/")
            assert path.is_file()
            assert path.suffix == ".txt"
            assert _is_git_tracked(path)


def test_quarantine_is_non_importable_reference_material() -> None:
    quarantine_root = Path("quarantine/qrwkv_xla")
    assert quarantine_root.is_dir()
    assert not (Path("src/radjax_tome/quarantine")).exists()
    assert not any(path.suffix == ".py" for path in quarantine_root.rglob("*"))
    assert (quarantine_root / "scripts/run_first_serious_burn.py.txt").is_file()
