from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from radjax_tome.audit.extraction_inventory import FileMatch, run_extraction_audit

CLOSURE_STATUSES = {
    "active_byte_identical",
    "active_function_equivalent",
    "active_behavior_equivalent",
    "quarantine_only",
    "student_bound",
    "contract_bound",
    "deprecated",
    "deferred",
    "waived",
    "missing",
    "unknown",
}

BLOCKING_FILE_STATUSES = {"missing", "unknown", "quarantine_only", "deferred", "waived"}
NON_BLOCKING_DISPOSITIONS = {"student_bound", "contract_bound", "deprecated"}
ACTIVE_EQUIVALENT_STATUSES = {
    "active_byte_identical",
    "active_function_equivalent",
    "active_behavior_equivalent",
}
OPTION_RE = re.compile(r"(?<!\\w)--[a-zA-Z0-9][a-zA-Z0-9_-]*")
KNOWN_ACTIVE_CLOSURE_PATHS = {
    "src/qrwkv_xla/artifacts/fingerprint.py": (
        "src/radjax_tome/fingerprint/artifacts.py",
        "src/radjax_tome/fingerprint/loader.py",
        "src/radjax_tome/fingerprint/validation.py",
        "src/radjax_tome/fingerprint/inspection.py",
    ),
}
CONTRACT_BOUND_PATH_MARKERS = (
    "docs/VOCAB_CONTRACT.md",
    "src/qrwkv_xla/contracts/",
)
STUDENT_BOUND_PATH_MARKERS = (
    "scripts/run_distill_stage.py",
    "scripts/tpu_distill_smoke.py",
    "scripts/train_student_smoke.py",
)


@dataclass(frozen=True)
class FunctionMapRecord:
    old_symbol: str
    old_symbol_type: str
    new_symbol: str | None
    new_path: str | None
    parity_status: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClosureFileRecord:
    old_path: str
    old_sha256: str | None
    old_role: str
    closure_status: str
    active_new_paths: tuple[str, ...] = ()
    quarantine_paths: tuple[str, ...] = ()
    function_map: tuple[FunctionMapRecord, ...] = ()
    tests_mapped: tuple[str, ...] = ()
    waiver_id: str | None = None
    reason: str = ""
    blocks_spec3: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_new_paths"] = list(self.active_new_paths)
        payload["quarantine_paths"] = list(self.quarantine_paths)
        payload["function_map"] = [item.to_dict() for item in self.function_map]
        payload["tests_mapped"] = list(self.tests_mapped)
        return payload


@dataclass(frozen=True)
class SymbolRecord:
    old_path: str
    old_symbol: str
    old_symbol_type: str
    closure_status: str
    new_symbol: str | None
    new_path: str | None
    evidence: str
    blocks_spec3: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CliRecord:
    old_cli: str
    new_cli: str | None
    old_help_status: str
    new_help_status: str
    old_arguments: tuple[str, ...]
    new_arguments: tuple[str, ...]
    removed_arguments: tuple[str, ...]
    added_arguments: tuple[str, ...]
    behavior_status: str
    reason: str
    blocks_spec3: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "old_arguments",
            "new_arguments",
            "removed_arguments",
            "added_arguments",
        ):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class TestRecord:
    old_test: str
    old_test_purpose: str
    active_new_tests: tuple[str, ...]
    quarantine_reference_only: bool
    status: str
    reason: str
    blocks_spec3: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["active_new_tests"] = list(self.active_new_tests)
        return payload


@dataclass(frozen=True)
class ClosureAudit:
    summary: dict[str, Any]
    spec3_gate: dict[str, Any]
    file_records: tuple[ClosureFileRecord, ...]
    symbol_records: tuple[SymbolRecord, ...]
    cli_records: tuple[CliRecord, ...]
    test_records: tuple[TestRecord, ...]
    ab_results: dict[str, Any]
    validation_results: dict[str, Any]
    waivers: tuple[dict[str, Any], ...]
    blockers: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "spec3_gate": self.spec3_gate,
            "file_records": [item.to_dict() for item in self.file_records],
            "symbol_records": [item.to_dict() for item in self.symbol_records],
            "cli_records": [item.to_dict() for item in self.cli_records],
            "test_records": [item.to_dict() for item in self.test_records],
            "ab_results": self.ab_results,
            "validation_results": self.validation_results,
            "waivers": list(self.waivers),
            "blockers": list(self.blockers),
        }


def run_closure_audit(
    old_repo: str | Path,
    new_repo: str | Path,
    *,
    ab_summary: str | Path | None = None,
) -> ClosureAudit:
    old_root = Path(old_repo).resolve()
    new_root = Path(new_repo).resolve()
    extraction = run_extraction_audit(old_root, new_root)
    ledger = _load_surgery_ledger(new_root)
    ledger_by_source = {entry["source_old_path"]: entry for entry in ledger}
    symbol_index = _build_symbol_index(new_root)

    file_records: list[ClosureFileRecord] = []
    symbol_records: list[SymbolRecord] = []
    for match in extraction.file_matches:
        file_record, symbols = _close_file(
            match,
            old_root=old_root,
            new_root=new_root,
            ledger_entry=ledger_by_source.get(match.old_path),
            symbol_index=symbol_index,
        )
        file_records.append(file_record)
        symbol_records.extend(symbols)

    cli_records = tuple(
        _close_cli(record, old_root=old_root, new_root=new_root)
        for record in file_records
        if record.old_path.startswith("scripts/")
    )
    test_records = tuple(
        _close_test(record)
        for record in file_records
        if record.old_path.startswith("tests/")
    )
    ledger_verification = _verify_ledger(new_root, ledger)
    ab_results = _read_ab_results(new_root, ab_summary=ab_summary)
    validation_results = {
        "ledger": ledger_verification,
        "import_boundary": _import_boundary_result(),
        "forbidden_import_grep": _forbidden_import_grep(new_root),
    }
    blockers = tuple(
        _blockers(
            file_records=file_records,
            symbol_records=symbol_records,
            cli_records=cli_records,
            test_records=test_records,
            ab_results=ab_results,
            validation_results=validation_results,
        )
    )
    summary = _summary(
        file_records=file_records,
        symbol_records=symbol_records,
        test_records=test_records,
        blockers=blockers,
        ab_results=ab_results,
        validation_results=validation_results,
    )
    spec3_gate = {
        "allowed": not blockers,
        "verdict": (
            "Spec 3 may proceed to Contract-valid Tome emission and cover_page.json."
            if not blockers
            else "Spec 3 blocked."
        ),
        "strict_requirements": {
            "missing_producer_files": summary["file_closure_counts"].get("missing", 0),
            "unknown_producer_files": summary["file_closure_counts"].get("unknown", 0),
            "missing_producer_symbols": summary["symbol_closure_counts"].get(
                "missing", 0
            ),
            "unknown_producer_symbols": summary["symbol_closure_counts"].get(
                "unknown", 0
            ),
            "blocking_quarantine_only_files": sum(
                1
                for record in file_records
                if record.closure_status == "quarantine_only" and record.blocks_spec3
            ),
            "blocking_deferred_items": sum(
                1
                for record in file_records
                if record.closure_status == "deferred" and record.blocks_spec3
            ),
            "blocking_waivers": sum(
                1
                for record in file_records
                if record.closure_status == "waived" and record.blocks_spec3
            ),
            "ab_parity_passes": ab_results.get("status") == "pass",
            "no_forbidden_imports": validation_results["import_boundary"]["ok"],
        },
    }
    return ClosureAudit(
        summary=summary,
        spec3_gate=spec3_gate,
        file_records=tuple(file_records),
        symbol_records=tuple(symbol_records),
        cli_records=cli_records,
        test_records=test_records,
        ab_results=ab_results,
        validation_results=validation_results,
        waivers=tuple(
            record.to_dict()
            for record in file_records
            if record.closure_status in {"waived", "deferred", "deprecated"}
        ),
        blockers=blockers,
    )


def write_closure_audit(
    audit: ClosureAudit,
    *,
    output_json: str | Path,
    output_md: str | Path,
    overwrite: bool = False,
) -> None:
    json_path = Path(output_json)
    md_path = Path(output_md)
    for path in (json_path, md_path):
        if path.exists() and not overwrite:
            raise ValueError(f"refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_closure_markdown(audit), encoding="utf-8")


def render_closure_markdown(audit: ClosureAudit) -> str:
    summary = audit.summary
    gate = audit.spec3_gate
    lines = [
        "# Tome Generator Closure Audit",
        "",
        "## Executive Verdict",
        "",
        gate["verdict"],
        "",
        f"Spec 3 allowed: `{gate['allowed']}`",
        "",
        "This is an adversarial closure report. Quarantine references, path-name",
        "matches, and passing tests are not counted as active migrated behavior",
        "unless the closure record has active paths and explicit evidence.",
        "",
        "## Closure Metrics",
        "",
        "### Files",
        "",
        *_markdown_counts(summary["file_closure_counts"]),
        "",
        "### Symbols",
        "",
        *_markdown_counts(summary["symbol_closure_counts"]),
        "",
        "### Tests",
        "",
        *_markdown_counts(summary["test_closure_counts"]),
        "",
        "## A/B Parity",
        "",
        f"- status: `{audit.ab_results.get('status')}`",
        f"- cases: `{audit.ab_results.get('cases')}`",
        f"- source: `{audit.ab_results.get('source')}`",
        "",
        "## Function And Symbol Parity Summary",
        "",
        (
            "No missing/unknown producer symbols were found."
            if not _symbol_blockers(audit.symbol_records)
            else "Missing or unknown producer symbols remain; see blockers below."
        ),
        "",
        "## CLI Parity Summary",
        "",
        *_cli_summary_lines(audit.cli_records),
        "",
        "## Test Parity Summary",
        "",
        "Active test coverage is counted separately from quarantine test references.",
        *_test_summary_lines(audit.test_records),
        "",
        "## Quarantine Ledger Verification",
        "",
        f"- ok: `{audit.validation_results['ledger']['ok']}`",
        f"- entries: `{audit.validation_results['ledger']['entries']}`",
        f"- blockers: `{len(audit.validation_results['ledger']['blockers'])}`",
        "",
        "## Active Vs Quarantine Accounting",
        "",
        "A quarantine `.txt` file is evidence only. It is classified as",
        "`quarantine_only`, `deferred`, `waived`, `student_bound`,",
        "`contract_bound`, or `deprecated` unless the surgery ledger or direct",
        "active-path evidence maps it to tracked active RADJAX-Tome code.",
        "",
        "## Remaining Blockers Or Waivers",
        "",
        *_blocker_lines(audit.blockers),
        "",
        "## Exact Next Recommendation",
        "",
        _recommendation(audit),
        "",
    ]
    return "\n".join(lines)


def _close_file(
    match: FileMatch,
    *,
    old_root: Path,
    new_root: Path,
    ledger_entry: dict[str, Any] | None,
    symbol_index: dict[str, list[tuple[str, str]]],
) -> tuple[ClosureFileRecord, list[SymbolRecord]]:
    old_path = old_root / match.old_path
    old_sha = _sha256(old_path) if old_path.is_file() else None
    active_paths = tuple(
        path
        for path in _active_paths(match, ledger_entry)
        if (new_root / path).is_file()
    )
    quarantine_paths = tuple(
        path
        for path in _quarantine_paths(match, ledger_entry)
        if (new_root / path).is_file()
    )
    status, reason = _closure_status(
        match, ledger_entry, active_paths, quarantine_paths
    )
    symbols = _extract_symbols(old_path)
    function_map = tuple(
        _map_symbol(
            symbol,
            status=status,
            active_paths=active_paths,
            symbol_index=symbol_index,
        )
        for symbol in symbols
    )
    blocks = _blocks_spec3(match, status, ledger_entry)
    tests = tuple(_tests_mapped(match, ledger_entry, active_paths))
    symbol_records = [
        SymbolRecord(
            old_path=match.old_path,
            old_symbol=item.old_symbol,
            old_symbol_type=item.old_symbol_type,
            closure_status=_symbol_status(item.parity_status),
            new_symbol=item.new_symbol,
            new_path=item.new_path,
            evidence=item.evidence,
            blocks_spec3=blocks
            and _symbol_status(item.parity_status) in {"missing", "unknown"},
        )
        for item in function_map
    ]
    return (
        ClosureFileRecord(
            old_path=match.old_path,
            old_sha256=old_sha,
            old_role=match.old_role,
            closure_status=status,
            active_new_paths=active_paths,
            quarantine_paths=quarantine_paths,
            function_map=function_map,
            tests_mapped=tests,
            waiver_id=_waiver_id(ledger_entry),
            reason=reason,
            blocks_spec3=blocks,
        ),
        symbol_records,
    )


def _closure_status(
    match: FileMatch,
    ledger_entry: dict[str, Any] | None,
    active_paths: tuple[str, ...],
    quarantine_paths: tuple[str, ...],
) -> tuple[str, str]:
    if _path_has_marker(match.old_path, CONTRACT_BOUND_PATH_MARKERS):
        return (
            "contract_bound",
            "Contract compatibility and vocabulary ownership belongs in "
            "RADJAX-Contract.",
        )
    if _path_has_marker(match.old_path, STUDENT_BOUND_PATH_MARKERS):
        return (
            "student_bound",
            "Student training/runtime CLI belongs in RADJAX-Student, not Tome.",
        )
    if ledger_entry is not None:
        decision = str(ledger_entry["decision"])
        if decision in {"promoted", "split_promoted"} and active_paths:
            return "active_behavior_equivalent", str(ledger_entry["reason"])
        if decision == "belongs_student":
            return "student_bound", str(ledger_entry["reason"])
        if decision == "belongs_contract":
            return "contract_bound", str(ledger_entry["reason"])
        if decision == "deprecated":
            return "deprecated", str(ledger_entry["reason"])
        if decision == "deferred":
            return "deferred", str(ledger_entry["reason"])
        if decision == "waived":
            return "waived", str(ledger_entry["reason"])
        if decision == "kept_quarantined":
            return "quarantine_only", str(ledger_entry["reason"])
    if match.new_status == "missing":
        return "missing", match.notes
    if match.new_status.startswith("intentionally_omitted_contract"):
        return "contract_bound", match.notes
    if match.new_status.startswith("intentionally_omitted_student"):
        return "student_bound", match.notes
    if match.new_status.startswith("intentionally_omitted_deprecated"):
        return "deprecated", match.notes
    if active_paths and match.new_status in {"migrated", "merged_into_other_file"}:
        return "active_behavior_equivalent", match.notes
    if quarantine_paths and not active_paths:
        return "quarantine_only", match.notes
    return "unknown", match.notes


def _map_symbol(
    symbol: dict[str, str],
    *,
    status: str,
    active_paths: tuple[str, ...],
    symbol_index: dict[str, list[tuple[str, str]]],
) -> FunctionMapRecord:
    candidates = [
        (path, kind)
        for path, kind in symbol_index.get(symbol["name"], [])
        if path in active_paths
    ]
    if candidates:
        path, _kind = candidates[0]
        return FunctionMapRecord(
            old_symbol=symbol["name"],
            old_symbol_type=symbol["kind"],
            new_symbol=symbol["name"],
            new_path=path,
            parity_status="renamed_equivalent"
            if status == "active_behavior_equivalent"
            else "behavior_equivalent",
            evidence=f"active symbol present in {path}",
        )
    if status in ACTIVE_EQUIVALENT_STATUSES:
        return FunctionMapRecord(
            old_symbol=symbol["name"],
            old_symbol_type=symbol["kind"],
            new_symbol=None,
            new_path=active_paths[0] if active_paths else None,
            parity_status="unknown",
            evidence="file mapped to active behavior but symbol name is not present",
        )
    if status in NON_BLOCKING_DISPOSITIONS:
        return FunctionMapRecord(
            old_symbol=symbol["name"],
            old_symbol_type=symbol["kind"],
            new_symbol=None,
            new_path=None,
            parity_status=status,
            evidence=f"file classified as {status}",
        )
    if status in {"deferred", "waived", "quarantine_only"}:
        return FunctionMapRecord(
            old_symbol=symbol["name"],
            old_symbol_type=symbol["kind"],
            new_symbol=None,
            new_path=None,
            parity_status="not_migrated",
            evidence=f"file classified as {status}",
        )
    return FunctionMapRecord(
        old_symbol=symbol["name"],
        old_symbol_type=symbol["kind"],
        new_symbol=None,
        new_path=None,
        parity_status="unknown" if status == "unknown" else "not_migrated",
        evidence=f"file classified as {status}",
    )


def _close_cli(
    record: ClosureFileRecord, *, old_root: Path, new_root: Path
) -> CliRecord:
    old_help = _help_result(old_root / record.old_path, old_root)
    new_cli = next(
        (path for path in record.active_new_paths if path.startswith("scripts/")), None
    )
    new_help = (
        _help_result(new_root / new_cli, new_root) if new_cli else _missing_help()
    )
    old_args = tuple(sorted(OPTION_RE.findall(old_help["text"])))
    new_args = tuple(sorted(OPTION_RE.findall(new_help["text"])))
    removed = tuple(sorted(set(old_args) - set(new_args)))
    added = tuple(sorted(set(new_args) - set(old_args)))
    if record.closure_status in ACTIVE_EQUIVALENT_STATUSES and new_cli:
        status = "active_behavior_equivalent" if not removed else "unknown"
    else:
        status = record.closure_status
    return CliRecord(
        old_cli=record.old_path,
        new_cli=new_cli,
        old_help_status=old_help["status"],
        new_help_status=new_help["status"],
        old_arguments=old_args,
        new_arguments=new_args,
        removed_arguments=removed,
        added_arguments=added,
        behavior_status=status,
        reason=(
            "CLI help compared."
            if old_help["status"] == "pass" and new_help["status"] == "pass"
            else (
                "CLI help unavailable or active CLI absent; status follows "
                "closure record."
            )
        ),
        blocks_spec3=record.blocks_spec3 and status in {"missing", "unknown"},
    )


def _close_test(record: ClosureFileRecord) -> TestRecord:
    active_tests = tuple(
        path for path in record.tests_mapped if path.startswith("tests/")
    )
    quarantine_only = bool(record.quarantine_paths) and not active_tests
    if active_tests:
        status = "active_equivalent"
    elif record.closure_status in {"student_bound", "contract_bound", "deprecated"}:
        status = record.closure_status
    elif quarantine_only:
        status = "quarantine_only"
    else:
        status = "missing" if record.closure_status == "missing" else "unknown"
    return TestRecord(
        old_test=record.old_path,
        old_test_purpose=record.old_role,
        active_new_tests=active_tests,
        quarantine_reference_only=quarantine_only,
        status=status,
        reason=record.reason,
        blocks_spec3=record.blocks_spec3
        and status in {"missing", "unknown", "quarantine_only"},
    )


def _verify_ledger(new_root: Path, ledger: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    for entry in ledger:
        if not entry.get("decision"):
            blockers.append(f"{entry.get('old_path')} missing decision")
        if not entry.get("reason"):
            blockers.append(f"{entry.get('old_path')} missing reason")
        decision = entry.get("decision")
        active_paths = entry.get("active_new_paths", [])
        if decision in {"promoted", "split_promoted"} and not active_paths:
            blockers.append(f"{entry['old_path']} promoted without active paths")
        for value in active_paths:
            path = new_root / value
            if not path.is_file():
                blockers.append(f"active path missing: {value}")
            elif not _is_git_tracked(new_root, value):
                blockers.append(f"active path not tracked: {value}")
        for value in entry.get("tests_added", []):
            path = new_root / value
            if not path.is_file():
                blockers.append(f"test path missing: {value}")
            elif not _is_git_tracked(new_root, value):
                blockers.append(f"test path not tracked: {value}")
        if decision in {"kept_quarantined", "deferred", "waived"}:
            if "blocks_spec3_after_this_phase" not in entry:
                blockers.append(f"{entry['old_path']} missing Spec 3 impact")
    return {"ok": not blockers, "entries": len(ledger), "blockers": blockers}


def _read_ab_results(
    new_root: Path, *, ab_summary: str | Path | None
) -> dict[str, Any]:
    path = (
        Path(ab_summary)
        if ab_summary
        else new_root / "artifacts/ab_teacher_textbook/ab_summary.json"
    )
    if not path.is_absolute():
        path = new_root / path
    if not path.is_file():
        return {"status": "missing", "cases": 0, "source": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": str(payload.get("status", "")),
        "cases": len(payload.get("cases", ())),
        "source": str(path),
        "failed_cases": [
            item.get("case_id")
            for item in payload.get("cases", ())
            if item.get("status") != "pass"
        ],
    }


def _import_boundary_result() -> dict[str, Any]:
    import radjax_tome  # noqa: F401

    bad = [
        name
        for name in sys.modules
        if name.startswith("qrwkv_xla")
        or name.startswith("quarantine")
        or name.startswith("jax")
        or name.startswith("torch")
        or name.startswith("transformers")
    ]
    return {"ok": not bad, "bad_imports": bad}


def _forbidden_import_grep(new_root: Path) -> dict[str, Any]:
    forbidden_roots = {"jax", "torch", "transformers"}
    hits: list[str] = []
    for root_name in ("src", "scripts", "tests"):
        root = new_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                root_import = _import_root(node)
                if root_import in forbidden_roots:
                    hits.append(
                        f"{path.relative_to(new_root)}:{node.lineno}:{root_import}"
                    )
    lazy_allowed = [
        hit for hit in hits if "corpora/tokenizer.py" in hit or "hf_causal_lm.py" in hit
    ]
    unexpected = [hit for hit in hits if hit not in lazy_allowed]
    return {"ok": not unexpected, "hits": hits, "unexpected_hits": unexpected}


def _blockers(
    *,
    file_records: list[ClosureFileRecord],
    symbol_records: list[SymbolRecord],
    cli_records: tuple[CliRecord, ...],
    test_records: tuple[TestRecord, ...],
    ab_results: dict[str, Any],
    validation_results: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for record in file_records:
        if record.blocks_spec3:
            blockers.append(
                {
                    "kind": "file",
                    "path": record.old_path,
                    "status": record.closure_status,
                    "reason": record.reason,
                }
            )
    for record in symbol_records:
        if record.blocks_spec3:
            blockers.append(
                {
                    "kind": "symbol",
                    "path": record.old_path,
                    "symbol": record.old_symbol,
                    "status": record.closure_status,
                    "reason": record.evidence,
                }
            )
    for record in cli_records:
        if record.blocks_spec3:
            blockers.append(
                {
                    "kind": "cli",
                    "path": record.old_cli,
                    "status": record.behavior_status,
                    "reason": record.reason,
                }
            )
    for record in test_records:
        if record.blocks_spec3:
            blockers.append(
                {
                    "kind": "test",
                    "path": record.old_test,
                    "status": record.status,
                    "reason": record.reason,
                }
            )
    if ab_results.get("status") != "pass":
        blockers.append(
            {
                "kind": "ab_parity",
                "path": str(ab_results.get("source")),
                "status": str(ab_results.get("status")),
                "reason": "A/B parity did not pass.",
            }
        )
    for key in ("ledger", "import_boundary", "forbidden_import_grep"):
        if not validation_results[key]["ok"]:
            blockers.append(
                {
                    "kind": "validation",
                    "path": key,
                    "status": "fail",
                    "reason": json.dumps(validation_results[key], sort_keys=True),
                }
            )
    return blockers


def _summary(
    *,
    file_records: list[ClosureFileRecord],
    symbol_records: list[SymbolRecord],
    test_records: tuple[TestRecord, ...],
    blockers: tuple[dict[str, Any], ...],
    ab_results: dict[str, Any],
    validation_results: dict[str, Any],
) -> dict[str, Any]:
    file_counts = Counter(record.closure_status for record in file_records)
    symbol_counts = Counter(record.closure_status for record in symbol_records)
    test_counts = Counter(record.status for record in test_records)
    return {
        "old_producer_relevant_files_total": len(file_records),
        "file_closure_counts": dict(sorted(file_counts.items())),
        "old_producer_symbols_total": len(symbol_records),
        "symbol_closure_counts": dict(sorted(symbol_counts.items())),
        "old_producer_tests_total": len(test_records),
        "test_closure_counts": dict(sorted(test_counts.items())),
        "spec3_blockers": len(blockers),
        "non_blocking_waivers": sum(
            1
            for record in file_records
            if record.closure_status in {"waived", "deferred", "deprecated"}
            and not record.blocks_spec3
        ),
        "blocking_waivers": sum(
            1
            for record in file_records
            if record.closure_status == "waived" and record.blocks_spec3
        ),
        "ab_parity_status": ab_results.get("status"),
        "audit_rerun_status": "complete",
        "ci_equivalent_validation_status": (
            "pass"
            if validation_results["ledger"]["ok"]
            and validation_results["import_boundary"]["ok"]
            and validation_results["forbidden_import_grep"]["ok"]
            else "fail"
        ),
    }


def _extract_symbols(path: Path) -> tuple[dict[str, str], ...]:
    if path.suffix != ".py" or not path.is_file():
        return ()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return ()
    symbols: list[dict[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            symbols.append({"name": node.name, "kind": "function"})
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            symbols.append({"name": node.name, "kind": "class"})
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append({"name": target.id, "kind": "constant"})
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id.isupper():
                symbols.append({"name": node.target.id, "kind": "constant"})
    return tuple(symbols)


def _build_symbol_index(new_root: Path) -> dict[str, list[tuple[str, str]]]:
    index: dict[str, list[tuple[str, str]]] = {}
    for root_name in ("src/radjax_tome", "scripts", "tests"):
        for path in (new_root / root_name).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(new_root).as_posix()
            for symbol in _extract_symbols(path):
                index.setdefault(symbol["name"], []).append((relative, symbol["kind"]))
    return index


def _active_paths(match: FileMatch, ledger_entry: dict[str, Any] | None) -> list[str]:
    paths = [
        path for path in match.new_path_or_paths if not path.startswith("quarantine/")
    ]
    paths.extend(KNOWN_ACTIVE_CLOSURE_PATHS.get(match.old_path, ()))
    if ledger_entry is not None:
        paths.extend(str(path) for path in ledger_entry.get("active_new_paths", ()))
    return sorted(dict.fromkeys(paths))


def _quarantine_paths(
    match: FileMatch, ledger_entry: dict[str, Any] | None
) -> list[str]:
    paths = [path for path in match.new_path_or_paths if path.startswith("quarantine/")]
    if ledger_entry is not None:
        paths.append(str(ledger_entry["old_path"]))
    return sorted(dict.fromkeys(paths))


def _path_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    return any(marker in path for marker in markers)


def _import_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        roots = {alias.name.split(".")[0] for alias in node.names}
        return next(iter(roots)) if len(roots) == 1 else None
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module.split(".")[0]
    return None


def _tests_mapped(
    match: FileMatch,
    ledger_entry: dict[str, Any] | None,
    active_paths: tuple[str, ...],
) -> list[str]:
    paths = [path for path in active_paths if path.startswith("tests/")]
    if ledger_entry is not None:
        paths.extend(str(path) for path in ledger_entry.get("tests_added", ()))
    if match.old_role == "producer_test":
        paths.extend(path for path in active_paths if path.startswith("tests/"))
    return sorted(dict.fromkeys(paths))


def _blocks_spec3(
    match: FileMatch, status: str, ledger_entry: dict[str, Any] | None
) -> bool:
    if ledger_entry is not None:
        return bool(ledger_entry.get("blocks_spec3_after_this_phase", False))
    if status in {"missing", "unknown"} and match.old_role in {
        "producer_core",
        "producer_validation",
        "producer_cli",
        "producer_capture",
        "mixed_producer_consumer",
    }:
        return True
    return False


def _symbol_status(parity_status: str) -> str:
    if parity_status in {"byte_identical", "ast_equivalent", "renamed_equivalent"}:
        return "active_function_equivalent"
    if parity_status == "behavior_equivalent":
        return "active_behavior_equivalent"
    if parity_status in {"student_bound", "contract_bound", "deprecated"}:
        return parity_status
    if parity_status == "not_migrated":
        return "missing"
    return "unknown"


def _waiver_id(ledger_entry: dict[str, Any] | None) -> str | None:
    if ledger_entry is None:
        return None
    if ledger_entry.get("decision") in {"waived", "deferred", "deprecated"}:
        return f"surgery:{ledger_entry['decision']}:{ledger_entry['source_old_path']}"
    return None


def _load_surgery_ledger(new_root: Path) -> list[dict[str, Any]]:
    path = new_root / "docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(item) for item in payload.get("entries", ())]


def _help_result(path: Path, cwd: Path) -> dict[str, str]:
    if not path:
        return _missing_help()
    if not path.is_file():
        return _missing_help()
    env = {**os.environ, "PYTHONPATH": str(cwd / "src")}
    try:
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "text": str(exc)}
    text = result.stdout + result.stderr
    return {"status": "pass" if result.returncode == 0 else "fail", "text": text}


def _missing_help() -> dict[str, str]:
    return {"status": "missing", "text": ""}


def _is_git_tracked(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _symbol_blockers(records: tuple[SymbolRecord, ...]) -> list[SymbolRecord]:
    return [record for record in records if record.blocks_spec3]


def _markdown_counts(counts: dict[str, int]) -> list[str]:
    return [f"- `{key}`: {counts[key]}" for key in sorted(counts)]


def _cli_summary_lines(records: tuple[CliRecord, ...]) -> list[str]:
    counts = Counter(record.behavior_status for record in records)
    lines = _markdown_counts(dict(counts))
    blocking = [record for record in records if record.blocks_spec3]
    lines.append(f"- blocking CLI records: {len(blocking)}")
    return lines


def _test_summary_lines(records: tuple[TestRecord, ...]) -> list[str]:
    counts = Counter(record.status for record in records)
    return _markdown_counts(dict(counts))


def _blocker_lines(blockers: tuple[dict[str, Any], ...]) -> list[str]:
    if not blockers:
        return ["- None."]
    lines: list[str] = []
    for item in blockers[:200]:
        symbol = f" symbol=`{item['symbol']}`" if item.get("symbol") else ""
        lines.append(
            "- "
            + f"{item['kind']}: `{item.get('path', '')}`{symbol} "
            + f"status=`{item['status']}` reason={item['reason']}"
        )
    return lines


def _recommendation(audit: ClosureAudit) -> str:
    if audit.spec3_gate["allowed"]:
        return "Proceed to Spec 3 Contract-valid Tome emission."
    return (
        "Do not start Spec 3. Resolve or explicitly waive every listed blocking "
        "file, symbol, CLI, test, A/B, and validation item first."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adversarial closure audit for qrwkv-xla to RADJAX-Tome."
    )
    parser.add_argument("--old-repo", type=Path, required=True)
    parser.add_argument("--new-repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--ab-summary", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-on-gate-blocked", action="store_true")
    args = parser.parse_args(argv)

    audit = run_closure_audit(
        args.old_repo,
        args.new_repo,
        ab_summary=args.ab_summary,
    )
    write_closure_audit(
        audit,
        output_json=args.output_json,
        output_md=args.output_md,
        overwrite=args.overwrite,
    )
    print(
        "status=complete "
        f"files={audit.summary['old_producer_relevant_files_total']} "
        f"symbols={audit.summary['old_producer_symbols_total']} "
        f"blockers={audit.summary['spec3_blockers']} "
        f"spec3_allowed={audit.spec3_gate['allowed']} "
        f"output_json={args.output_json}"
    )
    if args.fail_on_gate_blocked and not audit.spec3_gate["allowed"]:
        return 1
    return 0
