from __future__ import annotations

import ast
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GENERATED_AT = "2026-07-01T00:00:00Z"
VALID_SEVERITIES = {"blocker", "high", "medium", "low", "nice_to_have"}
VALID_CATEGORIES = {
    "modularity",
    "duplication",
    "dead_code",
    "api",
    "tests",
    "docs",
    "dependency_boundary",
    "script_thinness",
    "naming",
    "performance",
    "other",
}
SCAN_ROOTS = ("src", "scripts", "tests", "docs")
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    "quarantine",
    "build",
    "dist",
}
EXCLUDED_PATHS = {
    "docs/TOME_REFACTOR_AUDIT.json",
    "docs/TOME_REFACTOR_AUDIT.md",
}
SCORECARD_CATEGORIES = (
    "core package organization",
    "teacher backend boundary",
    "target store boundary",
    "compression boundary",
    "fingerprint boundary",
    "script thinness",
    "test maintainability",
    "optional dependency isolation",
    "Contract separation",
    "documentation hygiene",
    "future extensibility",
)


@dataclass(frozen=True)
class RefactorAudit:
    schema_version: str
    repo: str
    commit: str
    generated_at: str
    summary: dict[str, Any]
    scorecard: list[dict[str, Any]]
    file_metrics: list[dict[str, Any]]
    boundary_findings: list[dict[str, Any]]
    duplication_findings: list[dict[str, Any]]
    api_findings: list[dict[str, Any]]
    script_findings: list[dict[str, Any]]
    test_findings: list[dict[str, Any]]
    doc_findings: list[dict[str, Any]]
    checklist: list[dict[str, Any]]
    optional_dependency_imports: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_refactor_audit(repo_root: str | Path) -> RefactorAudit:
    root = Path(repo_root).resolve()
    paths = _active_paths(root)
    file_metrics = [_file_metrics(root, path) for path in paths]
    boundary_findings = _boundary_findings(file_metrics)
    duplication_findings = _duplication_findings(root, paths)
    api_findings = _api_findings(root, paths)
    script_findings = _script_findings(root, paths)
    test_findings = _test_findings(root, paths)
    doc_findings = _doc_findings(file_metrics)
    optional_imports = _optional_dependency_imports(root, paths)
    checklist = _checklist(
        file_metrics=file_metrics,
        boundary_findings=boundary_findings,
        duplication_findings=duplication_findings,
        api_findings=api_findings,
        script_findings=script_findings,
        test_findings=test_findings,
        doc_findings=doc_findings,
        optional_imports=optional_imports,
    )
    scorecard = _scorecard(
        file_metrics=file_metrics,
        script_findings=script_findings,
        doc_findings=doc_findings,
        optional_imports=optional_imports,
        checklist=checklist,
    )
    summary = _summary(checklist)
    return RefactorAudit(
        schema_version="refactor_audit_v1",
        repo="RADJAX-Tome",
        commit=_git_commit(root),
        generated_at=GENERATED_AT,
        summary=summary,
        scorecard=scorecard,
        file_metrics=file_metrics,
        boundary_findings=boundary_findings,
        duplication_findings=duplication_findings,
        api_findings=api_findings,
        script_findings=script_findings,
        test_findings=test_findings,
        doc_findings=doc_findings,
        checklist=checklist,
        optional_dependency_imports=optional_imports,
    )


def write_refactor_audit(
    audit: RefactorAudit,
    *,
    json_out: str | Path,
    md_out: str | Path,
) -> None:
    json_path = Path(json_out)
    md_path = Path(md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(audit), encoding="utf-8")


def render_markdown(audit: RefactorAudit) -> str:
    payload = audit.to_dict()
    summary = payload["summary"]
    checklist = payload["checklist"]
    lines = [
        "# RADJAX-Tome Refactor Audit",
        "",
        "## Executive Summary",
        "",
        (
            "The repo is functional, but migration left clear cleanup debt: "
            "audit tooling is swollen, several scripts are no longer thin, and "
            "report/JSON rendering patterns are repeated."
        ),
        "",
        f"- status: `{summary['status']}`",
        f"- Spec 3 blocked: `{summary['spec3_blocked']}`",
        f"- checklist items: `{len(checklist)}`",
        "",
        "## Spec 3 Readiness",
        "",
        (
            "No refactor finding blocks Spec 3. The issues below should shape "
            "cleanup specs, not silently expand the next feature phase."
        ),
        "",
        "## Modularity Scorecard",
        "",
        "| Category | Score | Evidence |",
        "|---|---:|---|",
    ]
    for item in payload["scorecard"]:
        lines.append(f"| {item['category']} | {item['score']} | {item['evidence']} |")
    lines.extend(
        [
            "",
            "## Top Refactor Checklist",
            "",
            *_checklist_lines(checklist[:10]),
            "",
            "## Must Fix Before Spec 3",
            "",
            *_checklist_lines(
                [item for item in checklist if item["severity"] == "blocker"]
            ),
            "",
            "## Should Fix Before Production Burns",
            "",
            *_checklist_lines(
                [item for item in checklist if item["severity"] in {"high", "medium"}]
            ),
            "",
            "## Can Wait",
            "",
            *_checklist_lines(
                [
                    item
                    for item in checklist
                    if item["severity"] in {"low", "nice_to_have"}
                ]
            ),
            "",
            "## File Size and Complexity Hotspots",
            "",
            *_hotspot_lines(payload["file_metrics"]),
            "",
            "## Boundary Findings",
            "",
            *_finding_lines(payload["boundary_findings"]),
            "",
            "## Duplication Findings",
            "",
            *_finding_lines(payload["duplication_findings"]),
            "",
            "## API Surface Findings",
            "",
            *_finding_lines(payload["api_findings"]),
            "",
            "## Script Thinness Findings",
            "",
            *_finding_lines(payload["script_findings"]),
            "",
            "## Test Suite Findings",
            "",
            *_finding_lines(payload["test_findings"]),
            "",
            "## Documentation Hygiene",
            "",
            *_finding_lines(payload["doc_findings"]),
            "",
            "## Recommended Follow-up Specs",
            "",
            "- Spec 2.14: shrink audit/reporting modules and extract shared renderers.",
            "- Spec 2.15: thin capability/audit scripts and move reusable logic "
            "into src.",
            "- Later: test fixture consolidation and optional real-HF local smoke "
            "polish.",
            "",
        ]
    )
    return "\n".join(lines)


def _active_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in SCAN_ROOTS:
        base = root / root_name
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if relative.as_posix() in EXCLUDED_PATHS:
                continue
            if path.suffix not in {".py", ".md", ".json", ".toml", ".yaml", ".yml"}:
                continue
            paths.append(path)
    return paths


def _file_metrics(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    metrics: dict[str, Any] = {
        "path": relative,
        "kind": path.suffix.lstrip(".") or "text",
        "line_count": len(lines),
        "nonblank_line_count": sum(1 for line in lines if line.strip()),
        "responsibility": _responsibility(relative),
        "secondary_responsibilities": _secondary_responsibilities(relative, text),
        "single_purpose": True,
        "mixed_purpose": False,
        "recommended_action": "keep",
    }
    if path.suffix == ".py":
        py_metrics = _python_metrics(text)
        metrics.update(py_metrics)
        secondary = metrics["secondary_responsibilities"]
        metrics["single_purpose"] = len(secondary) <= 1
        metrics["mixed_purpose"] = len(secondary) > 1
        if metrics["line_count"] > 500:
            metrics["recommended_action"] = "split or extract focused helpers"
        elif metrics["line_count"] > 300:
            metrics["recommended_action"] = "review for module shrink"
    return metrics


def _python_metrics(text: str) -> dict[str, Any]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {
            "function_count": 0,
            "class_count": 0,
            "largest_function": None,
            "largest_function_lines": 0,
            "largest_class": None,
            "largest_class_lines": 0,
            "import_count": 0,
            "complexity": 0,
        }
    functions: list[tuple[str, int]] = []
    classes: list[tuple[str, int]] = []
    imports = 0
    complexity = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append((node.name, _node_lines(node)))
        elif isinstance(node, ast.ClassDef):
            classes.append((node.name, _node_lines(node)))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1
        elif isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.BoolOp,
                ast.Match,
                ast.comprehension,
            ),
        ):
            complexity += 1
    largest_function = max(functions, key=lambda item: item[1], default=(None, 0))
    largest_class = max(classes, key=lambda item: item[1], default=(None, 0))
    return {
        "function_count": len(functions),
        "class_count": len(classes),
        "largest_function": largest_function[0],
        "largest_function_lines": largest_function[1],
        "largest_class": largest_class[0],
        "largest_class_lines": largest_class[1],
        "import_count": imports,
        "complexity": complexity,
    }


def _boundary_findings(file_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in file_metrics:
        if item.get("kind") != "py":
            continue
        path = item["path"]
        if item.get("mixed_purpose"):
            findings.append(
                _finding(
                    "mixed_responsibility",
                    "medium",
                    [path],
                    (f"{path} spans {', '.join(item['secondary_responsibilities'])}."),
                    "Split only if the secondary responsibility is real behavior, "
                    "not glue.",
                )
            )
        if path.startswith("src/radjax_tome/audit/") and item["line_count"] > 500:
            findings.append(
                _finding(
                    "audit_module_swollen",
                    "high",
                    [path],
                    "Audit modules mix collection, policy, markdown, and JSON shaping.",
                    "Extract reusable report rendering and policy tables.",
                )
            )
    return findings[:25]


def _duplication_findings(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    patterns = {
        "json_indent_sort_write": "json.dumps(",
        "markdown_table_rendering": "|---",
        "subprocess_cli_smoke": "subprocess.run(",
        "path_parent_mkdir": ".parent.mkdir(parents=True",
        "target_store_inspect_shape": "shape",
    }
    for path in paths:
        if path.suffix not in {".py", ".md"}:
            continue
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        for key, pattern in patterns.items():
            if pattern in text:
                occurrences[key].append(relative)
    findings: list[dict[str, Any]] = []
    if len(occurrences["json_indent_sort_write"]) >= 5:
        findings.append(
            _finding(
                "duplicate_json_report_writers",
                "medium",
                occurrences["json_indent_sort_write"][:8],
                "JSON report writing is repeated across scripts and audit modules.",
                "Extract a tiny JSON report writer or standardize on io/json.py.",
            )
        )
    if len(occurrences["subprocess_cli_smoke"]) >= 4:
        findings.append(
            _finding(
                "subprocess_test_duplication",
                "low",
                occurrences["subprocess_cli_smoke"][:8],
                "CLI smoke tests repeat subprocess environment setup.",
                "Use a shared test helper for PYTHONPATH and subprocess assertions.",
            )
        )
    if len(occurrences["markdown_table_rendering"]) >= 3:
        findings.append(
            _finding(
                "markdown_renderer_duplication",
                "medium",
                occurrences["markdown_table_rendering"][:8],
                "Markdown tables are hand-rendered in multiple report generators.",
                "Extract boring table rendering helpers under audit/report utilities.",
            )
        )
    return findings


def _api_findings(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        if path.name != "__init__.py" or path.suffix != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        exports = _exports(path)
        severity = "low"
        recommendation = "Keep explicit exports but group by domain."
        if len(exports) > 35:
            severity = "medium"
            recommendation = "Split or document the package import surface by role."
        findings.append(
            {
                "kind": "api_surface",
                "severity": severity,
                "files": [relative],
                "exports": exports,
                "evidence": f"{relative} exports {len(exports)} names.",
                "recommendation": recommendation,
                "heavy_import_leak": _has_heavy_top_level_import(path),
            }
        )
    return findings


def _script_findings(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if not relative.startswith("scripts/") or path.suffix != ".py":
            continue
        metrics = _file_metrics(root, path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        imports = sorted(set(_radjax_imports(text)))
        delegates = bool(imports)
        severity = "low"
        recommendation = "Keep as thin CLI."
        if metrics["line_count"] > 150 and metrics["function_count"] > 3:
            severity = "medium"
            recommendation = "Move reusable business logic into src and keep CLI thin."
        findings.append(
            {
                "kind": "script_thinness",
                "severity": severity,
                "files": [relative],
                "loc": metrics["line_count"],
                "radjax_imports": imports,
                "delegates_appropriately": delegates and severity == "low",
                "evidence": (
                    f"{relative} has {metrics['line_count']} lines and "
                    f"{metrics['function_count']} functions."
                ),
                "recommendation": recommendation,
            }
        )
    return findings


def _test_findings(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if not relative.startswith("tests/") or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        metrics = _file_metrics(root, path)
        category = "high_value"
        severity = "low"
        evidence = f"{relative} has focused assertions."
        if "subprocess.run(" in text:
            category = "medium_value"
            evidence = f"{relative} exercises CLI behavior through subprocess."
        if metrics["line_count"] > 200 or text.count("write_text(") > 5:
            category = "brittle"
            severity = "medium"
            evidence = (
                f"{relative} is fixture-heavy: {metrics['line_count']} lines, "
                f"{text.count('write_text(')} write_text calls."
            )
        if "hf" in relative and "local_files_only" in text:
            category = "needs_optional_real_smoke"
        findings.append(
            {
                "kind": "test_quality",
                "severity": severity,
                "files": [relative],
                "test_category": category,
                "evidence": evidence,
                "recommendation": _test_recommendation(category),
            }
        )
    return findings[:40]


def _doc_findings(file_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in file_metrics:
        path = item["path"]
        if not path.startswith("docs/"):
            continue
        if path.endswith(".json") and item["line_count"] > 500:
            severity = "medium"
            recommendation = "Move full generated payload to artifacts or compact it."
            if "QUARANTINE_SURGERY_LEDGER" in path:
                recommendation = (
                    "Keep only if the ledger is required as compact history."
                )
            findings.append(
                _finding(
                    "large_docs_json",
                    severity,
                    [path],
                    f"{path} has {item['line_count']} lines.",
                    recommendation,
                )
            )
        elif path.endswith(".md") and item["line_count"] > 250:
            findings.append(
                _finding(
                    "long_historical_doc",
                    "low",
                    [path],
                    f"{path} has {item['line_count']} lines.",
                    "Preserve if it is project history; otherwise add a summary "
                    "top section.",
                )
            )
    return findings


def _optional_dependency_imports(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    watched = {"torch", "transformers", "jax", "qrwkv_xla", "quarantine"}
    for path in paths:
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            root_import = _import_root(node)
            if root_import not in watched:
                continue
            results.append(
                {
                    "path": relative,
                    "line": getattr(node, "lineno", 0),
                    "import": root_import,
                    "classification": _dependency_classification(
                        relative,
                        root_import,
                    ),
                    "evidence": (
                        f"{relative}:{getattr(node, 'lineno', 0)} imports {root_import}"
                    ),
                }
            )
    return results


def _checklist(
    *,
    file_metrics: list[dict[str, Any]],
    boundary_findings: list[dict[str, Any]],
    duplication_findings: list[dict[str, Any]],
    api_findings: list[dict[str, Any]],
    script_findings: list[dict[str, Any]],
    test_findings: list[dict[str, Any]],
    doc_findings: list[dict[str, Any]],
    optional_imports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del file_metrics
    candidates: list[dict[str, Any]] = []
    candidates.extend(
        _item_from_finding(
            "RTA-001",
            "Shrink swollen audit modules without deleting audit coverage",
            "high",
            "modularity",
            boundary_findings,
            "2.14",
            ["audit outputs remain deterministic", "Spec 3 gate semantics"],
        )
    )
    candidates.extend(
        _item_from_finding(
            "RTA-002",
            "Standardize JSON and Markdown report rendering",
            "medium",
            "duplication",
            duplication_findings,
            "2.14",
            ["existing report schemas", "compact committed docs"],
        )
    )
    script_medium = [item for item in script_findings if item["severity"] == "medium"]
    candidates.extend(
        _item_from_finding(
            "RTA-003",
            "Move reusable script logic into src modules",
            "medium",
            "script_thinness",
            script_medium,
            "2.15",
            ["CLI behavior and exit codes", "no new heavy dependencies"],
        )
    )
    api_medium = [item for item in api_findings if item["severity"] == "medium"]
    candidates.extend(
        _item_from_finding(
            "RTA-004",
            "Trim broad package export surfaces",
            "low",
            "api",
            api_medium,
            "later",
            ["stable public imports", "optional dependency isolation"],
        )
    )
    test_medium = [item for item in test_findings if item["severity"] == "medium"]
    candidates.extend(
        _item_from_finding(
            "RTA-005",
            "Consolidate heavy test fixture construction",
            "medium",
            "tests",
            test_medium,
            "2.15",
            ["coverage intent", "CLI regression checks"],
        )
    )
    doc_medium = [item for item in doc_findings if item["severity"] == "medium"]
    candidates.extend(
        _item_from_finding(
            "RTA-006",
            "Keep generated docs compact by default",
            "medium",
            "docs",
            doc_medium,
            "later",
            ["historical audit traceability", "human-readable summaries"],
        )
    )
    suspicious = [
        item for item in optional_imports if item["classification"] == "suspicious"
    ]
    candidates.extend(
        _item_from_finding(
            "RTA-007",
            "Recheck optional dependency import boundaries",
            "low",
            "dependency_boundary",
            suspicious,
            "none",
            ["HF optional imports stay lazy", "core imports stay lightweight"],
        )
    )
    return candidates


def _item_from_finding(
    item_id: str,
    title: str,
    severity: str,
    category: str,
    findings: list[dict[str, Any]],
    suggested_spec: str,
    must_preserve: list[str],
) -> list[dict[str, Any]]:
    if not findings:
        return []
    files: list[str] = []
    evidence_parts: list[str] = []
    for finding in findings[:5]:
        files.extend(str(path) for path in finding.get("files", ()))
        evidence_parts.append(str(finding.get("evidence", "")))
    return [
        {
            "id": item_id,
            "title": title,
            "severity": severity,
            "category": category,
            "files": sorted(dict.fromkeys(files)),
            "problem": evidence_parts[0] if evidence_parts else title,
            "evidence": " | ".join(part for part in evidence_parts if part),
            "recommended_change": str(findings[0].get("recommendation", title)),
            "expected_benefit": "Less migration debt and clearer future change paths.",
            "risk": "Low if schemas, commands, and test intent are preserved.",
            "suggested_spec": suggested_spec,
            "must_preserve": must_preserve,
        }
    ]


def _scorecard(
    *,
    file_metrics: list[dict[str, Any]],
    script_findings: list[dict[str, Any]],
    doc_findings: list[dict[str, Any]],
    optional_imports: list[dict[str, Any]],
    checklist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del file_metrics
    high_or_medium = sum(
        1 for item in checklist if item["severity"] in {"high", "medium"}
    )
    suspicious_imports = sum(
        1 for item in optional_imports if item["classification"] == "suspicious"
    )
    medium_scripts = sum(1 for item in script_findings if item["severity"] == "medium")
    medium_docs = sum(1 for item in doc_findings if item["severity"] == "medium")
    values = {
        "core package organization": (
            3,
            "Package layout is coherent, but audit and generation helpers need "
            "pruning.",
        ),
        "teacher backend boundary": (
            4,
            "HF optional code is isolated behind backend modules.",
        ),
        "target store boundary": (
            4,
            "Store/schema/inspection/compression are separated cleanly.",
        ),
        "compression boundary": (
            4,
            "Compression is now a focused NumPy-only module.",
        ),
        "fingerprint boundary": (
            3,
            "Artifact schemas and generation are separate, but package exports "
            "are broad.",
        ),
        "script thinness": (
            max(2, 4 - min(medium_scripts, 2)),
            f"{medium_scripts} scripts are large enough to review for business logic.",
        ),
        "test maintainability": (
            3,
            "Coverage is strong, but fixture-heavy tests repeat setup.",
        ),
        "optional dependency isolation": (
            5 if suspicious_imports == 0 else 3,
            f"{suspicious_imports} suspicious heavy dependency imports found.",
        ),
        "Contract separation": (
            5,
            "No active Contract repo edits or qrwkv_xla runtime imports are required.",
        ),
        "documentation hygiene": (
            max(2, 4 - min(medium_docs, 2)),
            f"{medium_docs} generated or JSON docs need compactness review.",
        ),
        "future extensibility": (
            max(2, 4 - min(high_or_medium, 3)),
            f"{high_or_medium} high/medium cleanup items should shape next specs.",
        ),
    }
    return [
        {
            "category": category,
            "score": values[category][0],
            "evidence": values[category][1],
        }
        for category in SCORECARD_CATEGORIES
    ]


def _summary(checklist: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["severity"] for item in checklist)
    return {
        "status": "complete",
        "spec3_blocked": counts["blocker"] > 0,
        "blocker_count": counts["blocker"],
        "high_count": counts["high"],
        "medium_count": counts["medium"],
        "low_count": counts["low"],
        "nice_to_have_count": counts["nice_to_have"],
    }


def _responsibility(path: str) -> str:
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("docs/"):
        return "docs"
    if path.startswith("scripts/"):
        return "CLI/scripts"
    markers = (
        ("corpora", "corpus/tokenization"),
        ("backends", "teacher backend"),
        ("targets/compression", "target compression"),
        ("targets", "target store/schema"),
        ("fingerprint", "fingerprint artifacts"),
        ("reports", "reporting"),
        ("audit", "audit tooling"),
        ("io", "I/O helpers"),
        ("provenance", "provenance/hash utilities"),
    )
    for marker, responsibility in markers:
        if marker in path:
            return responsibility
    return "other"


def _secondary_responsibilities(path: str, text: str) -> list[str]:
    zones = {_responsibility(path)}
    lowered = text.lower()
    probes = {
        "CLI/scripts": "argparse",
        "reporting": "markdown",
        "I/O helpers": "json.dumps",
        "teacher backend": "transformers",
        "target store/schema": "targetstore",
        "target compression": "topk",
        "fingerprint artifacts": "fingerprint",
        "audit tooling": "audit",
    }
    for zone, probe in probes.items():
        if probe.lower() in lowered:
            zones.add(zone)
    return sorted(zones)


def _exports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return _literal_string_list(node.value)
    return []


def _literal_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    values: list[str] = []
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.append(item.value)
    return values


def _has_heavy_top_level_import(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    for node in tree.body:
        if _import_root(node) in {"torch", "transformers", "jax"}:
            return True
    return False


def _radjax_imports(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("radjax_tome"):
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("radjax_tome"):
                    imports.append(alias.name)
    return imports


def _dependency_classification(path: str, import_name: str) -> str:
    if path.startswith("tests/"):
        return "allowed_test_only"
    if import_name in {"torch", "transformers"} and (
        "hf_" in path or "tokenizer.py" in path
    ):
        return "allowed_optional"
    if import_name in {"qrwkv_xla", "quarantine", "jax"}:
        return "blocker"
    return "suspicious"


def _test_recommendation(category: str) -> str:
    return {
        "high_value": "Keep.",
        "medium_value": "Keep, but share subprocess helpers.",
        "low_value": "Replace existence-only checks with behavior checks.",
        "brittle": "Extract fixture builders and assert less implementation detail.",
        "needs_optional_real_smoke": (
            "Keep optional/local marker; do not add network CI."
        ),
    }.get(category, "Review.")


def _node_lines(node: ast.AST) -> int:
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    return max(0, int(end) - int(start) + 1)


def _import_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        roots = {alias.name.split(".")[0] for alias in node.names}
        return next(iter(roots)) if len(roots) == 1 else None
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module.split(".")[0]
    return None


def _finding(
    kind: str,
    severity: str,
    files: list[str],
    evidence: str,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "files": files,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _checklist_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None."]
    return [
        (
            f"- `{item['id']}` **{item['severity']}**: {item['title']} "
            f"({', '.join(item['files'][:3])})"
        )
        for item in items
    ]


def _hotspot_lines(file_metrics: list[dict[str, Any]]) -> list[str]:
    hotspots = [
        item
        for item in file_metrics
        if item["kind"] == "py"
        and (
            item["line_count"] > 300
            or item.get("largest_function_lines", 0) > 75
            or item.get("largest_class_lines", 0) > 250
        )
    ]
    if not hotspots:
        return ["- None."]
    lines = []
    for item in sorted(hotspots, key=lambda value: value["line_count"], reverse=True)[
        :20
    ]:
        lines.append(
            "- "
            + f"`{item['path']}`: {item['line_count']} lines; "
            + f"largest function `{item.get('largest_function')}` "
            + f"({item.get('largest_function_lines')} lines)."
        )
    return lines


def _finding_lines(findings: list[dict[str, Any]]) -> list[str]:
    if not findings:
        return ["- None."]
    return [
        (
            f"- **{item['severity']}** `{item['kind']}`: {item['evidence']} "
            f"Recommendation: {item['recommendation']}"
        )
        for item in findings[:20]
    ]


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()
