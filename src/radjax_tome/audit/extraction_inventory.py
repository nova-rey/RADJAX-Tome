from __future__ import annotations

import ast
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PRODUCER_KEYWORDS = (
    "teacher_textbook",
    "TeacherTextbook",
    "build_teacher_textbook",
    "validate_teacher_textbook",
    "teacher_target",
    "TeacherTargetStore",
    "target_store",
    "export_teacher_targets",
    "inspect_targets",
    "target_type",
    "dense_logits",
    "topk_with_tail",
    "cascaded_soft_labels",
    "bucket_edges",
    "bucket_mass",
    "bucket_mean_logp",
    "bucket_shape",
    "teacher-hf",
    "hf_causal_lm",
    "real_teacher",
    "local_files_only",
    "fingerprint capture",
    "capture_fingerprint",
    "FingerprintCapture",
    "stat_bands",
    "exemplar_reservoir",
    "capture_summary",
    "prompt corpus",
    "corpora",
    "source_text",
    "example_id",
    "split_manifest",
)

PRODUCER_CORE_KEYWORDS = (
    "TeacherTextbook",
    "TeacherTargetStore",
    "TargetStoreMetadata",
    "build_teacher_textbook",
    "validate_teacher_textbook",
    "dense_logits",
    "topk_with_tail",
    "cascaded_soft_labels",
    "bucket_edges",
    "bucket_mass",
    "bucket_mean_logp",
    "hf_causal_lm",
    "local_files_only",
)

CAPTURE_KEYWORDS = (
    "capture_fingerprint",
    "FingerprintCapture",
    "capture_summary",
    "stat_bands",
    "mode discovery",
    "exemplar_reservoir",
    "real_teacher",
    "teacher logits",
)

STUDENT_ONLY_KEYWORDS = (
    "student backend",
    "optimizer",
    "checkpoint",
    "train step",
    "run_distill_stage",
    "Pallas",
    "JAX runtime",
    "two-cycle",
    "training loop",
    "held-out checkpoint",
)

CONTRACT_KEYWORDS = (
    "vocab contract",
    "artifact schema",
    "validation schema",
    "manifest schema",
    "compatibility",
)

OLD_TO_NEW_PATH_HINTS = (
    (
        "src/qrwkv_xla/artifacts/teacher_textbook_builder.py",
        "src/radjax_tome/builder/teacher_textbook.py",
    ),
    (
        "src/qrwkv_xla/artifacts/teacher_textbook.py",
        "src/radjax_tome/builder/teacher_textbook.py",
    ),
    (
        "src/qrwkv_xla/artifacts/cascaded_soft_labels.py",
        "src/radjax_tome/builder/cascaded_soft_labels.py",
    ),
    ("src/qrwkv_xla/targets/store.py", "src/radjax_tome/targets/store.py"),
    ("src/qrwkv_xla/targets/schema.py", "src/radjax_tome/targets/schema.py"),
    (
        "src/qrwkv_xla/targets/consumption.py",
        "src/radjax_tome/targets/consumption.py",
    ),
    (
        "src/qrwkv_xla/targets/multishard.py",
        "src/radjax_tome/targets/multishard.py",
    ),
    (
        "src/qrwkv_xla/generation/tokenizer.py",
        "src/radjax_tome/corpora/tokenizer.py",
    ),
    (
        "src/qrwkv_xla/teachers/backend.py",
        "src/radjax_tome/backends/base.py",
    ),
    (
        "src/qrwkv_xla/teachers/emission.py",
        "src/radjax_tome/backends/emission.py",
    ),
    (
        "src/qrwkv_xla/teachers/synthetic.py",
        "src/radjax_tome/backends/synthetic.py",
    ),
    ("src/qrwkv_xla/teachers/hf.py", "src/radjax_tome/backends/hf_causal_lm.py"),
    ("scripts/build_teacher_textbook.py", "scripts/build_teacher_textbook.py"),
    ("scripts/validate_teacher_textbook.py", "scripts/validate_teacher_textbook.py"),
    ("scripts/export_teacher_targets.py", "scripts/export_teacher_targets.py"),
    ("scripts/inspect_targets.py", "scripts/inspect_targets.py"),
    ("scripts/tokenize_corpus.py", "scripts/tokenize_corpus.py"),
    ("scripts/validate_pipeline.py", "scripts/validate_producer_pipeline.py"),
    ("tests/test_multishard_target_store.py", "tests/test_target_core_migration.py"),
    ("tests/test_offline_target_consumption.py", "tests/test_target_core_migration.py"),
    ("tests/test_teacher_backend_emission.py", "tests/test_target_core_migration.py"),
    ("tests/test_export_teacher_targets_cli.py", "tests/test_target_core_migration.py"),
    ("tests/test_smoke_tokenizer.py", "tests/test_corpus_tokenization.py"),
    ("tests/test_tokenize_corpus_script.py", "tests/test_corpus_tokenization.py"),
    ("tests/test_tokenizer_registry.py", "tests/test_corpus_tokenization.py"),
)

HIGH_RISK_ROLES = {
    "producer_core",
    "producer_validation",
    "producer_cli",
    "producer_capture",
    "mixed_producer_consumer",
}

COMMON_SYMBOLS = {
    "main",
    "to_dict",
    "from_dict",
    "load",
    "save",
    "run",
    "validate",
}


@dataclass(frozen=True)
class FileInventoryItem:
    path: str
    role: str
    matched_keywords: tuple[str, ...]
    symbols: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FileMatch:
    old_path: str
    old_role: str
    old_symbols_or_entrypoints: tuple[str, ...]
    new_status: str
    new_path_or_paths: tuple[str, ...]
    migration_confidence: str
    notes: str
    action_required: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    old_repo: str
    new_repo: str
    old_repo_ok: bool
    new_repo_ok: bool
    candidate_keywords: tuple[str, ...]
    old_candidates: tuple[FileInventoryItem, ...]
    new_candidates: tuple[FileInventoryItem, ...]
    file_matches: tuple[FileMatch, ...]
    cli_inventory: tuple[dict[str, Any], ...]
    test_inventory: tuple[dict[str, Any], ...]
    docs_fixtures_inventory: tuple[dict[str, Any], ...]
    symbol_inventory: dict[str, Any]
    summary: dict[str, int]
    high_risk_missing_items: tuple[dict[str, Any], ...]
    blockers_before_spec3: tuple[str, ...]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_repo": self.old_repo,
            "new_repo": self.new_repo,
            "old_repo_ok": self.old_repo_ok,
            "new_repo_ok": self.new_repo_ok,
            "candidate_keywords": list(self.candidate_keywords),
            "old_candidates": [item.to_dict() for item in self.old_candidates],
            "new_candidates": [item.to_dict() for item in self.new_candidates],
            "file_matches": [item.to_dict() for item in self.file_matches],
            "cli_inventory": list(self.cli_inventory),
            "test_inventory": list(self.test_inventory),
            "docs_fixtures_inventory": list(self.docs_fixtures_inventory),
            "symbol_inventory": self.symbol_inventory,
            "summary": dict(self.summary),
            "high_risk_missing_items": list(self.high_risk_missing_items),
            "blockers_before_spec3": list(self.blockers_before_spec3),
            "recommendation": self.recommendation,
        }


def run_extraction_audit(old_repo: str | Path, new_repo: str | Path) -> AuditReport:
    old_root = Path(old_repo).resolve()
    new_root = Path(new_repo).resolve()
    old_candidates = tuple(scan_repo(old_root))
    new_candidates = tuple(scan_repo(new_root))
    matches = tuple(match_old_to_new(old_candidates, new_candidates))
    summary = _build_summary(old_candidates, matches)
    high_risk = tuple(_high_risk_items(matches))
    blockers = tuple(item["old_path"] for item in high_risk)
    recommendation = _recommendation(blockers)
    return AuditReport(
        old_repo=str(old_root),
        new_repo=str(new_root),
        old_repo_ok=_appears_to_be_old_repo(old_root),
        new_repo_ok=_appears_to_be_new_repo(new_root),
        candidate_keywords=PRODUCER_KEYWORDS,
        old_candidates=old_candidates,
        new_candidates=new_candidates,
        file_matches=matches,
        cli_inventory=tuple(_cli_inventory(matches)),
        test_inventory=tuple(_test_inventory(matches)),
        docs_fixtures_inventory=tuple(_docs_fixtures_inventory(matches)),
        symbol_inventory=_symbol_inventory(matches, new_candidates),
        summary=summary,
        high_risk_missing_items=high_risk,
        blockers_before_spec3=blockers,
        recommendation=recommendation,
    )


def scan_repo(root: Path) -> list[FileInventoryItem]:
    items: list[FileInventoryItem] = []
    for path in sorted(_iter_scannable_files(root)):
        relative = path.relative_to(root).as_posix()
        text = _read_text(path)
        haystack = f"{relative}\n{text}"
        matched = tuple(keyword for keyword in PRODUCER_KEYWORDS if keyword in haystack)
        role = classify_file(relative, text, matched)
        if matched or role not in {"unknown_needs_human_review", "student_only"}:
            items.append(
                FileInventoryItem(
                    path=relative,
                    role=role,
                    matched_keywords=matched,
                    symbols=tuple(extract_public_symbols(path, text)),
                )
            )
    return items


def classify_file(
    relative_path: str,
    text: str,
    matched_keywords: tuple[str, ...] = (),
) -> str:
    lower_path = relative_path.lower()
    lower_text = text.lower()
    combined = f"{lower_path}\n{lower_text}"

    if relative_path.startswith("tests/") and matched_keywords:
        return "producer_test"
    if relative_path.startswith("docs/") and matched_keywords:
        return "producer_doc"
    if _is_fixture_path(relative_path) and matched_keywords:
        return "producer_fixture"
    if lower_path.startswith("scripts/") and _contains_any(
        combined,
        (
            "teacher",
            "textbook",
            "target",
            "tome",
            "fingerprint",
            "export_teacher",
            "inspect_targets",
        ),
    ):
        return "producer_cli"
    has_capture = _contains_any(text, CAPTURE_KEYWORDS)
    has_student = _contains_any(text, STUDENT_ONLY_KEYWORDS)
    if has_capture and has_student:
        return "mixed_producer_consumer"
    if has_capture:
        return "producer_capture"
    if _contains_any(text, PRODUCER_CORE_KEYWORDS):
        if has_student:
            return "mixed_producer_consumer"
        if "validate" in lower_path or "validation" in lower_text:
            return "producer_validation"
        return "producer_core"
    if _contains_any(text, CONTRACT_KEYWORDS):
        return "contract_only"
    if has_student:
        return "student_only"
    if matched_keywords:
        return "unknown_needs_human_review"
    return "student_only"


def match_old_to_new(
    old_items: tuple[FileInventoryItem, ...],
    new_items: tuple[FileInventoryItem, ...],
) -> list[FileMatch]:
    new_by_path = {item.path: item for item in new_items}
    new_by_name: dict[str, list[FileInventoryItem]] = {}
    new_symbols: dict[str, list[FileInventoryItem]] = {}
    for item in new_items:
        new_by_name.setdefault(Path(item.path).name, []).append(item)
        for symbol in item.symbols:
            new_symbols.setdefault(symbol, []).append(item)

    matches: list[FileMatch] = []
    for old in old_items:
        if old.role == "student_only":
            continue
        hinted = _hinted_new_path(old.path, new_by_path)
        basename_matches = new_by_name.get(Path(old.path).name, [])
        symbol_matches = _symbol_matches(old, new_symbols)
        new_paths = _unique_paths([*hinted, *basename_matches, *symbol_matches])
        status, confidence, notes, action = _status_for_match(
            old, new_paths, new_by_path
        )
        matches.append(
            FileMatch(
                old_path=old.path,
                old_role=old.role,
                old_symbols_or_entrypoints=old.symbols,
                new_status=status,
                new_path_or_paths=tuple(item.path for item in new_paths),
                migration_confidence=confidence,
                notes=notes,
                action_required=action,
            )
        )
    return matches


def write_audit_reports(
    report: AuditReport, output_dir: str | Path, *, overwrite: bool = False
) -> None:
    root = Path(output_dir)
    if root.exists() and not overwrite:
        raise ValueError(f"audit output dir already exists: {root}")
    if root.exists():
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "extraction_audit.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "extraction_audit.md").write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: AuditReport) -> str:
    lines = [
        "# Tome Generator Extraction Audit",
        "",
        f"- old repo: `{report.old_repo}`",
        f"- new repo: `{report.new_repo}`",
        f"- old repo recognized: `{report.old_repo_ok}`",
        f"- new repo recognized: `{report.new_repo_ok}`",
        "",
        "## Summary",
        "",
    ]
    for key in sorted(report.summary):
        lines.append(f"- {key}: `{report.summary[key]}`")
    lines.extend(["", "## Recommendation", "", report.recommendation, ""])
    lines.extend(["## High-Risk Missing Or Partial Items", ""])
    if report.high_risk_missing_items:
        for item in report.high_risk_missing_items[:50]:
            detail = (
                f"- `{item['old_path']}`: {item['status']} ({item['role']}) - "
                f"{item['action_required']}"
            )
            lines.append(detail)
    else:
        lines.append("- None detected by the heuristic audit.")
    lines.extend(["", "## Producer Test Inventory", ""])
    for item in report.test_inventory[:80]:
        new_paths = ", ".join(item["new_paths"]) or "none"
        lines.append(f"- `{item['old_path']}`: {item['status']} -> {new_paths}")
    lines.extend(["", "## CLI Inventory", ""])
    for item in report.cli_inventory:
        lines.append(
            f"- `{item['old_cli']}`: producer_relevant=`{item['producer_relevant']}` "
            f"status=`{item['status']}` new=`{item['new_equivalent'] or 'none'}`"
        )
    lines.extend(["", "## File Matches", ""])
    for item in report.file_matches[:200]:
        lines.append(
            f"- `{item.old_path}` ({item.old_role}) -> `{item.new_status}` "
            f"{', '.join(item.new_path_or_paths) or 'none'}"
        )
    lines.append("")
    return "\n".join(lines)


def extract_public_symbols(path: Path, text: str) -> list[str]:
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef)
        ) and not node.name.startswith("_"):
            symbols.append(node.name)
    return symbols


def _status_for_match(
    old: FileInventoryItem,
    new_paths: list[FileInventoryItem],
    new_by_path: dict[str, FileInventoryItem],
) -> tuple[str, str, str, str]:
    if old.role == "contract_only":
        return (
            "intentionally_omitted_contract",
            "medium",
            "Contract-shaped surface belongs in RADJAX-Contract unless wrapped.",
            "Document waiver or migrate only if RADJAX-Tome needs a producer wrapper.",
        )
    if old.role == "student_only":
        return (
            "intentionally_omitted_student",
            "high",
            "Student-side implementation is outside Tome Generator scope.",
            "No RADJAX-Tome migration required.",
        )
    if _looks_deprecated(old.path):
        return (
            "intentionally_omitted_deprecated",
            "medium",
            "Historical/deprecated path naming detected.",
            "Confirm deprecation before ignoring.",
        )
    if not new_paths:
        return (
            "missing",
            "high" if old.role in HIGH_RISK_ROLES else "medium",
            "No path, basename, or public-symbol equivalent found in RADJAX-Tome.",
            "Review and either migrate, waive, or reclassify.",
        )
    if _is_hinted_full_match(old.path, new_paths):
        return (
            "migrated",
            "high",
            "Matched through known old-to-new path mapping.",
            "No immediate action.",
        )
    if _same_basename_match(old, new_paths):
        return (
            "migrated",
            "medium",
            "Matched by identical filename in a producer-relevant location.",
            "Confirm semantic equivalence if high risk.",
        )
    if _symbol_overlap(old, new_paths) >= max(1, min(3, len(_specific_symbols(old)))):
        return (
            "merged_into_other_file",
            "medium",
            "Public symbols appear in one or more RADJAX-Tome files.",
            "Confirm behavior and tests were ported.",
        )
    if Path(old.path).name in new_by_path:
        return (
            "migrated",
            "medium",
            "Matched by identical relative path.",
            "Confirm semantic equivalence if high risk.",
        )
    return (
        "partial",
        "low",
        "Only weak basename or keyword overlap found.",
        "Review manually before Spec 3.",
    )


def _build_summary(
    old_candidates: tuple[FileInventoryItem, ...],
    matches: tuple[FileMatch, ...],
) -> dict[str, int]:
    status_counts = Counter(match.new_status for match in matches)
    role_counts = Counter(item.role for item in old_candidates)
    producer_tests = [match for match in matches if match.old_role == "producer_test"]
    return {
        "total_old_candidate_files": len(old_candidates),
        "producer_relevant_old_files": len(matches),
        "migrated": status_counts["migrated"] + status_counts["merged_into_other_file"],
        "partial": status_counts["partial"],
        "missing": status_counts["missing"],
        "intentionally_omitted": sum(
            count
            for status, count in status_counts.items()
            if status.startswith("intentionally_omitted")
        ),
        "needs_human_review": status_counts["needs_human_review"],
        "producer_relevant_old_tests": len(producer_tests),
        "new_equivalent_tests": sum(
            1
            for match in producer_tests
            if match.new_status in {"migrated", "merged_into_other_file", "partial"}
        ),
        "missing_tests": sum(
            1 for match in producer_tests if match.new_status == "missing"
        ),
        "producer_core_files": role_counts["producer_core"],
        "producer_cli_files": role_counts["producer_cli"],
        "mixed_producer_consumer_files": role_counts["mixed_producer_consumer"],
    }


def _high_risk_items(matches: tuple[FileMatch, ...]) -> list[dict[str, Any]]:
    risky: list[dict[str, Any]] = []
    for match in matches:
        if match.old_role in HIGH_RISK_ROLES and match.new_status in {
            "missing",
            "partial",
            "needs_human_review",
        }:
            risky.append(
                {
                    "old_path": match.old_path,
                    "role": match.old_role,
                    "status": match.new_status,
                    "action_required": match.action_required,
                }
            )
    return risky


def _recommendation(blockers: tuple[str, ...]) -> str:
    if blockers:
        return (
            "Spec 3 should not proceed until the high-risk missing or partial "
            "producer-side gaps are migrated or explicitly waived."
        )
    return (
        "Spec 3 may proceed because the heuristic audit did not find high-risk "
        "producer-side migration blockers."
    )


def _cli_inventory(matches: tuple[FileMatch, ...]) -> list[dict[str, Any]]:
    return [
        {
            "old_cli": match.old_path,
            "producer_relevant": True,
            "new_equivalent": ", ".join(match.new_path_or_paths),
            "status": match.new_status,
            "reason": match.notes,
        }
        for match in matches
        if match.old_path.startswith("scripts/")
    ]


def _test_inventory(matches: tuple[FileMatch, ...]) -> list[dict[str, Any]]:
    return [
        {
            "old_path": match.old_path,
            "status": match.new_status,
            "new_paths": list(match.new_path_or_paths),
            "action_required": match.action_required,
        }
        for match in matches
        if match.old_role == "producer_test"
    ]


def _docs_fixtures_inventory(matches: tuple[FileMatch, ...]) -> list[dict[str, Any]]:
    return [
        {
            "old_path": match.old_path,
            "role": match.old_role,
            "status": match.new_status,
            "new_paths": list(match.new_path_or_paths),
        }
        for match in matches
        if match.old_role in {"producer_doc", "producer_fixture"}
    ]


def _symbol_inventory(
    matches: tuple[FileMatch, ...],
    new_candidates: tuple[FileInventoryItem, ...],
) -> dict[str, Any]:
    new_symbol_set = {symbol for item in new_candidates for symbol in item.symbols}
    high_value = sorted(
        {symbol for match in matches for symbol in match.old_symbols_or_entrypoints}
    )
    missing = [symbol for symbol in high_value if symbol not in new_symbol_set]
    return {
        "old_public_symbols": high_value,
        "new_public_symbols": sorted(new_symbol_set),
        "missing_old_symbols": missing,
    }


def _iter_scannable_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    ignored_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
    allowed_suffixes = {
        ".py",
        ".md",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".txt",
        ".toml",
    }
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix in allowed_suffixes or path.parent.name in {
            "corpora",
            "configs",
        }:
            files.append(path)
    return files


def _hinted_new_path(
    old_path: str, new_by_path: dict[str, FileInventoryItem]
) -> list[FileInventoryItem]:
    matches: list[FileInventoryItem] = []
    for old_hint, new_hint in OLD_TO_NEW_PATH_HINTS:
        if old_path == old_hint and new_hint in new_by_path:
            matches.append(new_by_path[new_hint])
    translated = old_path.replace("src/qrwkv_xla/", "src/radjax_tome/")
    if translated in new_by_path:
        matches.append(new_by_path[translated])
    return matches


def _symbol_matches(
    old: FileInventoryItem,
    new_symbols: dict[str, list[FileInventoryItem]],
) -> list[FileInventoryItem]:
    matches: list[FileInventoryItem] = []
    for symbol in _specific_symbols(old):
        matches.extend(new_symbols.get(symbol, []))
    return matches


def _unique_paths(items: list[FileInventoryItem]) -> list[FileInventoryItem]:
    seen: set[str] = set()
    unique: list[FileInventoryItem] = []
    for item in items:
        if item.path not in seen:
            seen.add(item.path)
            unique.append(item)
    return unique


def _symbol_overlap(old: FileInventoryItem, new_paths: list[FileInventoryItem]) -> int:
    new_symbols = {symbol for item in new_paths for symbol in item.symbols}
    return len(set(_specific_symbols(old)) & new_symbols)


def _specific_symbols(item: FileInventoryItem) -> tuple[str, ...]:
    return tuple(symbol for symbol in item.symbols if symbol not in COMMON_SYMBOLS)


def _same_basename_match(
    old: FileInventoryItem,
    new_paths: list[FileInventoryItem],
) -> bool:
    old_name = Path(old.path).name
    return any(Path(item.path).name == old_name for item in new_paths)


def _is_hinted_full_match(old_path: str, new_paths: list[FileInventoryItem]) -> bool:
    new_path_set = {item.path for item in new_paths}
    return any(
        old_path == old_hint and new_hint in new_path_set
        for old_hint, new_hint in OLD_TO_NEW_PATH_HINTS
    )


def _appears_to_be_old_repo(root: Path) -> bool:
    return (root / "src" / "qrwkv_xla").is_dir()


def _appears_to_be_new_repo(root: Path) -> bool:
    return (root / "src" / "radjax_tome").is_dir()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(keyword.lower() in lower_text for keyword in keywords)


def _is_fixture_path(path: str) -> bool:
    parts = set(Path(path).parts)
    return bool(parts & {"corpora", "configs", "fixtures", "testdata", "data"})


def _looks_deprecated(path: str) -> bool:
    lower = path.lower()
    return "deprecated" in lower or "legacy_unused" in lower
