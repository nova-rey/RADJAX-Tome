from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from radjax_tome.audit.closure import run_closure_audit, write_closure_audit

ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_SYMBOLS = {
    "PROBABILITY_LIKE_STATS",
    "TARGET_PAYLOAD_LEGACY_JSONL",
    "TARGET_PAYLOAD_PACKED_CORRIDOR_V1",
    "PACKED_TARGET_ARRAYS",
    "ValidationResult",
}


def _write_repo_pair(tmp_path: Path) -> tuple[Path, Path]:
    old = tmp_path / "old"
    new = tmp_path / "new"
    (old / "src/qrwkv_xla").mkdir(parents=True)
    (new / "src/radjax_tome").mkdir(parents=True)
    (old / "src/qrwkv_xla/__init__.py").write_text("", encoding="utf-8")
    (new / "src/radjax_tome/__init__.py").write_text("", encoding="utf-8")
    (old / "scripts").mkdir()
    (new / "scripts").mkdir()
    script = """
from __future__ import annotations

import argparse

TEACHER_TEXTBOOK_KIND = "TeacherTextbook"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""".lstrip()
    (old / "scripts/build_teacher_textbook.py").write_text(script, encoding="utf-8")
    (new / "scripts/build_teacher_textbook.py").write_text(script, encoding="utf-8")
    return old, new


def _write_fingerprint_repo_pair(
    tmp_path: Path, *, omit_symbol: str = ""
) -> tuple[Path, Path]:
    old = tmp_path / "old"
    new = tmp_path / "new"
    (old / "src/qrwkv_xla/artifacts").mkdir(parents=True)
    (new / "src/radjax_tome/fingerprint").mkdir(parents=True)
    (old / "src/qrwkv_xla/artifacts/fingerprint.py").write_text(
        """
from dataclasses import dataclass, field
from typing import Any

FingerprintCapture = object

PROBABILITY_LIKE_STATS = frozenset(
    {"top1_margin", "top8_mass", "top32_mass", "tail_mass"}
)
TARGET_PAYLOAD_LEGACY_JSONL = "legacy_jsonl"
TARGET_PAYLOAD_PACKED_CORRIDOR_V1 = "packed_corridor_v1"
PACKED_TARGET_ARRAYS = {
    "examples_input_ids": 2,
    "position_example_index": 1,
    "position": 1,
    "mode_id": 1,
    "weight": 1,
}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
""".lstrip(),
        encoding="utf-8",
    )
    active_lines = [
        "from dataclasses import dataclass, field",
        "from typing import Any",
        "",
    ]
    if omit_symbol != "PROBABILITY_LIKE_STATS":
        active_lines.extend(
            [
                "PROBABILITY_LIKE_STATS = frozenset(",
                '    ("top1_margin", "top8_mass", "top32_mass", "tail_mass")',
                ")",
            ]
        )
    if omit_symbol != "TARGET_PAYLOAD_LEGACY_JSONL":
        active_lines.append('TARGET_PAYLOAD_LEGACY_JSONL = "legacy_jsonl"')
    if omit_symbol != "TARGET_PAYLOAD_PACKED_CORRIDOR_V1":
        active_lines.append('TARGET_PAYLOAD_PACKED_CORRIDOR_V1 = "packed_corridor_v1"')
    if omit_symbol != "PACKED_TARGET_ARRAYS":
        active_lines.extend(
            [
                "PACKED_TARGET_ARRAYS = {",
                '    "examples_input_ids": 2,',
                '    "position_example_index": 1,',
                '    "position": 1,',
                '    "mode_id": 1,',
                '    "weight": 1,',
                "}",
            ]
        )
    if omit_symbol != "FingerprintValidationResult":
        active_lines.extend(
            [
                "",
                "@dataclass(frozen=True)",
                "class FingerprintValidationResult:",
                "    ok: bool",
                "    blockers: tuple[str, ...] = ()",
                "    warnings: tuple[str, ...] = ()",
                "    metadata: dict[str, Any] = field(default_factory=dict)",
            ]
        )
    (new / "src/radjax_tome/fingerprint/artifacts.py").write_text(
        "\n".join(active_lines) + "\n",
        encoding="utf-8",
    )
    return old, new


def test_closure_audit_allows_minimal_active_equivalence(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(
        json.dumps(
            {"status": "pass", "cases": [{"case_id": "unit", "status": "pass"}]}
        ),
        encoding="utf-8",
    )

    audit = run_closure_audit(old, new, ab_summary=ab_summary)

    assert audit.spec3_gate["allowed"] is True
    assert audit.summary["file_closure_counts"]["active_behavior_equivalent"] == 1
    assert audit.summary["symbol_closure_counts"]["active_function_equivalent"] == 2


def test_closure_audit_blocks_missing_producer_surface(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    (new / "scripts/build_teacher_textbook.py").unlink()
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    audit = run_closure_audit(old, new, ab_summary=ab_summary)

    assert audit.spec3_gate["allowed"] is False
    assert any(blocker["kind"] == "file" for blocker in audit.blockers)


def test_fingerprint_closure_symbols_map_to_active_tome_schema(tmp_path: Path) -> None:
    old, new = _write_fingerprint_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    audit = run_closure_audit(old, new, ab_summary=ab_summary)
    records = {
        record.old_symbol: record
        for record in audit.symbol_records
        if record.old_path == "src/qrwkv_xla/artifacts/fingerprint.py"
    }

    assert FINGERPRINT_SYMBOLS <= records.keys()
    assert records["ValidationResult"].new_symbol == "FingerprintValidationResult"
    for symbol in FINGERPRINT_SYMBOLS:
        assert records[symbol].closure_status == "active_function_equivalent"
        assert records[symbol].blocks_spec3 is False
    assert not [
        blocker
        for blocker in audit.blockers
        if blocker.get("path") == "src/qrwkv_xla/artifacts/fingerprint.py"
    ]


def test_fingerprint_closure_mapping_fails_when_active_target_disappears(
    tmp_path: Path,
) -> None:
    old, new = _write_fingerprint_repo_pair(
        tmp_path,
        omit_symbol="TARGET_PAYLOAD_PACKED_CORRIDOR_V1",
    )
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    audit = run_closure_audit(old, new, ab_summary=ab_summary)

    assert audit.spec3_gate["allowed"] is False
    assert any(
        blocker.get("kind") == "symbol"
        and blocker.get("symbol") == "TARGET_PAYLOAD_PACKED_CORRIDOR_V1"
        for blocker in audit.blockers
    )


def test_closure_script_writes_json_and_markdown(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    output_json = tmp_path / "closure.json"
    output_summary_json = tmp_path / "closure-summary.json"
    output_md = tmp_path / "closure.md"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_tome_generator_closure.py",
            "--old-repo",
            str(old),
            "--new-repo",
            str(new),
            "--ab-summary",
            str(ab_summary),
            "--output-json",
            str(output_json),
            "--output-summary-json",
            str(output_summary_json),
            "--output-md",
            str(output_md),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "status=complete" in result.stdout
    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]
    summary = json.loads(output_summary_json.read_text(encoding="utf-8"))
    assert summary["kind"] == "radjax_tome_closure_audit_summary"
    assert "file_records" not in summary
    assert len(output_summary_json.read_text(encoding="utf-8").splitlines()) < 1000
    assert "Spec 3 allowed" in output_md.read_text(encoding="utf-8")


def test_write_closure_audit_refuses_overwrite(tmp_path: Path) -> None:
    old, new = _write_repo_pair(tmp_path)
    ab_summary = tmp_path / "ab_summary.json"
    ab_summary.write_text(json.dumps({"status": "pass", "cases": []}), encoding="utf-8")
    audit = run_closure_audit(old, new, ab_summary=ab_summary)
    output_json = tmp_path / "closure.json"
    output_md = tmp_path / "closure.md"
    write_closure_audit(
        audit,
        output_json=output_json,
        output_md=output_md,
        overwrite=True,
    )

    try:
        write_closure_audit(
            audit,
            output_json=output_json,
            output_md=output_md,
            overwrite=False,
        )
    except ValueError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("write_closure_audit should refuse overwrite")


def test_committed_closure_summary_json_is_compact() -> None:
    path = ROOT / "docs/TOME_GENERATOR_CLOSURE_AUDIT.json"
    assert path.is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) < 1000
