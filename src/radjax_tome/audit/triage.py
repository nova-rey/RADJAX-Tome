from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

TRIAGE_BUCKETS = (
    "must_migrate_tome_before_spec3",
    "migrate_tome_before_full_burn",
    "migrate_tome_deferred",
    "belongs_contract",
    "belongs_student",
    "mixed_requires_split",
    "historical_deprecated",
    "duplicate_or_merged",
    "waive_with_reason",
    "needs_human_review",
)

TEST_BUCKETS = (
    "must_port_before_spec3",
    "must_port_with_associated_feature",
    "defer",
    "waive",
    "belongs_student",
    "belongs_contract",
    "historical_deprecated",
)

ROADMAP = (
    {
        "spec": "Spec 2.7",
        "title": "Producer Schemas, Stores, Validators, and Core Tests",
        "goal": (
            "Migrate or waive producer schemas, target stores, validators, "
            "export/inspection CLIs, and core tests required before cover-page work."
        ),
        "must_include_files_or_areas": (
            "teacher artifact schemas and manifests",
            "target-store and multishard producer helpers",
            "export/inspect/validate teacher target CLIs",
            "dense/top-k/cascaded producer tests",
        ),
        "must_include_tests": (
            "teacher target store tests",
            "top-k/tail textbook tests",
            "cascaded soft-label textbook tests",
            "export/inspect/validate CLI tests",
        ),
        "must_not_include": (
            "cover_page.json implementation",
            "student training loops",
            "bulk fingerprint migration",
        ),
        "acceptance_criteria": (
            "Spec 2.5 audit has no unwaived producer-core Spec 3 blockers",
            "legacy A/B parity still passes",
        ),
        "risk_level": "high",
    },
    {
        "spec": "Spec 2.8",
        "title": "Corpus, Source Identity, Real-Teacher/HF Producer Paths",
        "goal": (
            "Restore producer-side corpus/source identity, HF/local-files-only, "
            "and real-teacher capture entry points needed by Tome generation."
        ),
        "must_include_files_or_areas": (
            "prompt corpus loading and splitting",
            "source/example_id preservation",
            "HF teacher export and specimen smoke",
            "real-teacher producer capture boundaries",
        ),
        "must_include_tests": (
            "prompt corpus tests",
            "HF teacher backend/export tests",
            "real-teacher offline producer tests",
        ),
        "must_not_include": ("student runtime migration", "cover_page.json emission"),
        "acceptance_criteria": (
            "source identity is covered by tests",
            "HF/local-files-only producer path is tested or explicitly waived",
        ),
        "risk_level": "high",
    },
    {
        "spec": "Spec 2.9",
        "title": "Behavioral Fingerprint, Corridor, Exemplar Producer Artifacts",
        "goal": (
            "Split and migrate producer-side behavioral fingerprint, exemplar, "
            "corridor, and capture-summary artifact paths."
        ),
        "must_include_files_or_areas": (
            "fingerprint artifact schemas and validators",
            "exemplar target/exemplar artifact writers",
            "capture_summary.json style outputs",
            "stat bands and provenance metadata",
        ),
        "must_include_tests": (
            "fingerprint artifact validation tests",
            "fingerprint capture parity tests",
            "compressed exemplar tests",
        ),
        "must_not_include": ("student optimizer/checkpoint internals",),
        "acceptance_criteria": (
            "mixed producer/student files have explicit split destinations",
            "producer artifact tests are ported or waived",
        ),
        "risk_level": "high",
    },
    {
        "spec": "Spec 2.10",
        "title": "Audit Closure, A/B Expansion, and Explicit Waivers",
        "goal": (
            "Rerun extraction audit and A/B parity, close or explicitly waive "
            "every remaining blocker, and lock the Spec 3 entry gate."
        ),
        "must_include_files_or_areas": (
            "updated extraction audit",
            "updated migration map",
            "expanded A/B cases if migrated behavior requires them",
            "waiver register",
        ),
        "must_include_tests": (
            "audit/triage regression tests",
            "legacy A/B parity tests",
        ),
        "must_not_include": ("cover_page.json implementation",),
        "acceptance_criteria": (
            "Spec 3 gate passes",
            "no untriaged high-risk producer blockers remain",
        ),
        "risk_level": "medium",
    },
)


@dataclass(frozen=True)
class TriageItem:
    old_path: str
    old_role: str
    old_status: str
    bucket: str
    proposed_destination_repo: str
    proposed_destination_path_or_area: str
    migration_priority: str
    blocks_spec3: bool
    why_high_risk: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TestTriageItem:
    old_path: str
    test_bucket: str
    linked_migration_bucket: str
    blocks_spec3: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Spec3Gate:
    passed: bool
    high_risk_must_migrate_tome_before_spec3: int
    high_risk_blocking_mixed_requires_split: int
    missing_must_port_before_spec3_tests: int
    untriaged_high_risk_items: int
    legacy_ab_parity_required: bool
    extraction_audit_closure_required: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MigrationMap:
    source_audit_json: str
    audit_summary: dict[str, int]
    triage_summary: dict[str, int]
    high_risk_by_bucket: dict[str, int]
    missing_tests_by_bucket: dict[str, int]
    triage_items: tuple[TriageItem, ...]
    missing_test_triage: tuple[TestTriageItem, ...]
    roadmap: tuple[dict[str, Any], ...]
    spec3_gate: Spec3Gate
    open_questions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_audit_json": self.source_audit_json,
            "audit_summary": self.audit_summary,
            "triage_summary": self.triage_summary,
            "high_risk_by_bucket": self.high_risk_by_bucket,
            "missing_tests_by_bucket": self.missing_tests_by_bucket,
            "triage_items": [item.to_dict() for item in self.triage_items],
            "missing_test_triage": [
                item.to_dict() for item in self.missing_test_triage
            ],
            "roadmap": list(self.roadmap),
            "spec3_gate": self.spec3_gate.to_dict(),
            "open_questions": list(self.open_questions),
        }


def load_audit_json(path: str | Path) -> dict[str, Any]:
    audit_path = Path(path)
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    required = {
        "summary",
        "file_matches",
        "high_risk_missing_items",
        "test_inventory",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"audit JSON missing required top-level fields: {missing}")
    return payload


def build_migration_map(audit_json: str | Path) -> MigrationMap:
    audit_path = Path(audit_json)
    audit = load_audit_json(audit_path)
    actionable = [
        item
        for item in audit["file_matches"]
        if item.get("new_status") in {"missing", "partial", "needs_human_review"}
    ]
    high_risk_paths = {
        item["old_path"] for item in audit.get("high_risk_missing_items", [])
    }
    triage_items = tuple(
        triage_file_match(item, high_risk=item["old_path"] in high_risk_paths)
        for item in actionable
    )
    missing_test_triage = tuple(
        triage_missing_test(item)
        for item in audit["test_inventory"]
        if item.get("status") == "missing"
    )
    spec3_gate = compute_spec3_gate(triage_items, missing_test_triage)
    return MigrationMap(
        source_audit_json=str(audit_path),
        audit_summary={str(k): int(v) for k, v in audit["summary"].items()},
        triage_summary=_count_items_by_bucket(triage_items),
        high_risk_by_bucket=_count_items_by_bucket(
            item for item in triage_items if item.why_high_risk
        ),
        missing_tests_by_bucket=dict(
            Counter(item.test_bucket for item in missing_test_triage)
        ),
        triage_items=triage_items,
        missing_test_triage=missing_test_triage,
        roadmap=ROADMAP,
        spec3_gate=spec3_gate,
        open_questions=_open_questions(triage_items),
    )


def triage_file_match(match: dict[str, Any], *, high_risk: bool = False) -> TriageItem:
    old_path = str(match["old_path"])
    old_role = str(match["old_role"])
    old_status = str(match["new_status"])
    bucket, repo, area, priority, blocks, notes = _classify_item(
        old_path=old_path,
        old_role=old_role,
        old_status=old_status,
        new_paths=tuple(match.get("new_path_or_paths", ())),
    )
    why = ""
    if high_risk:
        why = (
            "Spec 2.5 marked this path as high risk because it is missing or "
            "partial producer-core, producer-CLI, producer-validation, "
            "producer-capture, or mixed producer/consumer surface."
        )
    return TriageItem(
        old_path=old_path,
        old_role=old_role,
        old_status=old_status,
        bucket=bucket,
        proposed_destination_repo=repo,
        proposed_destination_path_or_area=area,
        migration_priority=priority,
        blocks_spec3=blocks,
        why_high_risk=why,
        notes=notes,
    )


def triage_missing_test(test: dict[str, Any]) -> TestTriageItem:
    old_path = str(test["old_path"])
    linked_bucket = _linked_bucket_for_test(old_path)
    if linked_bucket == "belongs_student":
        test_bucket = "belongs_student"
        blocks = False
    elif linked_bucket == "belongs_contract":
        test_bucket = "belongs_contract"
        blocks = False
    elif _is_historical(old_path):
        test_bucket = "historical_deprecated"
        blocks = False
    elif linked_bucket == "must_migrate_tome_before_spec3":
        test_bucket = "must_port_before_spec3"
        blocks = True
    elif linked_bucket in {"mixed_requires_split", "migrate_tome_before_full_burn"}:
        test_bucket = "must_port_with_associated_feature"
        blocks = linked_bucket == "mixed_requires_split"
    else:
        test_bucket = "defer"
        blocks = False
    return TestTriageItem(
        old_path=old_path,
        test_bucket=test_bucket,
        linked_migration_bucket=linked_bucket,
        blocks_spec3=blocks,
        notes=_test_notes(old_path, linked_bucket),
    )


def compute_spec3_gate(
    triage_items: tuple[TriageItem, ...],
    missing_tests: tuple[TestTriageItem, ...],
) -> Spec3Gate:
    must = sum(
        1
        for item in triage_items
        if item.why_high_risk and item.bucket == "must_migrate_tome_before_spec3"
    )
    mixed = sum(
        1
        for item in triage_items
        if item.why_high_risk
        and item.bucket == "mixed_requires_split"
        and item.blocks_spec3
    )
    tests = sum(
        1
        for item in missing_tests
        if item.test_bucket == "must_port_before_spec3" and item.blocks_spec3
    )
    quarantined = sum(
        1
        for item in triage_items
        if item.why_high_risk
        and item.bucket == "duplicate_or_merged"
        and "quarantine/" in item.proposed_destination_path_or_area
    )
    untriaged = sum(
        1
        for item in triage_items
        if item.why_high_risk and item.bucket == "needs_human_review"
    )
    reasons: list[str] = []
    if must:
        reasons.append(
            f"{must} high-risk Tome producer items must migrate before Spec 3"
        )
    if mixed:
        reasons.append(f"{mixed} high-risk mixed producer/student items need splits")
    if tests:
        reasons.append(f"{tests} missing producer tests must port before Spec 3")
    if quarantined:
        reasons.append(
            f"{quarantined} high-risk producer items are quarantined for Spec 2.9"
        )
    if untriaged:
        reasons.append(f"{untriaged} high-risk items remain untriaged")
    reasons.append("legacy A/B parity must still pass after migration waves")
    reasons.append(
        "extraction audit must rerun with no untriaged producer-core blockers"
    )
    return Spec3Gate(
        passed=not (must or mixed or tests or quarantined or untriaged),
        high_risk_must_migrate_tome_before_spec3=must,
        high_risk_blocking_mixed_requires_split=mixed,
        missing_must_port_before_spec3_tests=tests,
        untriaged_high_risk_items=untriaged,
        legacy_ab_parity_required=True,
        extraction_audit_closure_required=True,
        reasons=tuple(reasons),
    )


def write_migration_map(
    migration_map: MigrationMap,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    root = Path(output_dir)
    if root.exists():
        if not overwrite:
            raise ValueError(f"migration map output dir already exists: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "migration_map.json").write_text(
        json.dumps(migration_map.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "migration_map.md").write_text(
        render_migration_map_markdown(migration_map),
        encoding="utf-8",
    )


def render_migration_map_markdown(migration_map: MigrationMap) -> str:
    lines = [
        "# Tome Generator Migration Map",
        "",
        "## Audit Summary",
        "",
        _summary_table(migration_map.audit_summary),
        "",
        "## Spec 3 Gate",
        "",
        f"Passed: `{migration_map.spec3_gate.passed}`",
        "",
    ]
    for reason in migration_map.spec3_gate.reasons:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Triage Summary",
            "",
            _summary_table(migration_map.triage_summary),
            "",
            "## High-Risk Blockers By Bucket",
            "",
            _summary_table(migration_map.high_risk_by_bucket),
            "",
            "## Missing Tests By Bucket",
            "",
            _summary_table(migration_map.missing_tests_by_bucket),
            "",
            "## Short-Term Roadmap",
            "",
        ]
    )
    for chunk in migration_map.roadmap:
        lines.extend(
            [
                f"### {chunk['spec']} - {chunk['title']}",
                "",
                f"Goal: {chunk['goal']}",
                "",
                "Must include:",
            ]
        )
        lines.extend(f"- {item}" for item in chunk["must_include_files_or_areas"])
        lines.append("")
        lines.append("Must include tests:")
        lines.extend(f"- {item}" for item in chunk["must_include_tests"])
        lines.append("")
        lines.append("Must not include:")
        lines.extend(f"- {item}" for item in chunk["must_not_include"])
        lines.append("")
        lines.append("Acceptance criteria:")
        lines.extend(f"- {item}" for item in chunk["acceptance_criteria"])
        lines.extend(["", f"Risk level: `{chunk['risk_level']}`", ""])
    lines.extend(["## High-Risk Path-Level Triage", ""])
    high_risk = [item for item in migration_map.triage_items if item.why_high_risk]
    for item in high_risk:
        lines.append(
            f"- `{item.old_path}` -> `{item.bucket}`; "
            f"repo=`{item.proposed_destination_repo}`; "
            f"priority=`{item.migration_priority}`; "
            f"blocks_spec3=`{item.blocks_spec3}`; "
            f"area={item.proposed_destination_path_or_area}"
        )
    lines.extend(["", "## Open Questions", ""])
    if migration_map.open_questions:
        lines.extend(f"- {item}" for item in migration_map.open_questions)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def emit_doc_summary(migration_map: MigrationMap, path: str | Path) -> None:
    Path(path).write_text(render_doc_summary(migration_map), encoding="utf-8")


def render_doc_summary(migration_map: MigrationMap) -> str:
    lines = [
        "# Tome Generator Migration Map",
        "",
        "## Summary",
        "",
        "Spec 3 is blocked. The extraction audit found a narrow TeacherTextbook "
        "migration, not a complete Tome Generator extraction.",
        "",
        _summary_table(migration_map.audit_summary),
        "",
        "## Short-Term Roadmap",
        "",
        "- Spec 2.5 - Extraction completeness audit. DONE.",
        "- Spec 2.6 - Audit triage and producer migration map. DONE.",
        "- Spec 2.7 - Migrate highest-priority producer "
        "schemas/stores/validators/tests. DONE.",
        "- Spec 2.8 - Bulk producer migration with quarantine. THIS SPEC.",
        "- Spec 2.9 - Surgical split of quarantined mixed producer/student files.",
        "- Spec 2.10 - Audit closure, A/B expansion, waivers, and Spec 3 gate check.",
        "- Spec 3 - Contract-valid Tome emission with cover_page.json, only after "
        "the gate passes.",
        "",
        "Previous micro-migration roadmap:",
        "",
        "- Spec 2.8 - Migrate real-teacher/HF/corpus/source-identity producer paths.",
        "- Spec 2.9 - Migrate behavioral fingerprint / corridor / exemplar "
        "producer artifact paths.",
        "- Spec 2.10 - Re-run extraction audit and A/B parity; reduce blockers "
        "to zero or explicit waivers.",
        "- Spec 3 - Only then implement Contract-valid Tome emission with "
        "cover_page.json.",
        "",
        "## Bucket Definitions",
        "",
    ]
    lines.extend(f"- `{bucket}`" for bucket in TRIAGE_BUCKETS)
    lines.extend(
        [
            "",
            "## Triage Summary Counts",
            "",
            _summary_table(migration_map.triage_summary),
            "",
            "## High-Risk Blocker Summary",
            "",
            _summary_table(migration_map.high_risk_by_bucket),
            "",
            "## Missing Test Summary",
            "",
            _summary_table(migration_map.missing_tests_by_bucket),
            "",
            "## Ordered Migration Chunks",
            "",
        ]
    )
    for chunk in migration_map.roadmap:
        lines.extend(
            [
                f"### {chunk['spec']} - {chunk['title']}",
                "",
                chunk["goal"],
                "",
                f"Risk: `{chunk['risk_level']}`",
                "",
                "Must include:",
            ]
        )
        lines.extend(f"- {item}" for item in chunk["must_include_files_or_areas"])
        lines.append("")
        lines.append("Acceptance criteria:")
        lines.extend(f"- {item}" for item in chunk["acceptance_criteria"])
        lines.append("")
    lines.extend(
        [
            "## Spec 3 Gate",
            "",
            f"Passed: `{migration_map.spec3_gate.passed}`",
            "",
        ]
    )
    lines.extend(f"- {reason}" for reason in migration_map.spec3_gate.reasons)
    lines.extend(["", "## Open Questions / Human Review", ""])
    if migration_map.open_questions:
        lines.extend(f"- {item}" for item in migration_map.open_questions)
    else:
        lines.append("- No `needs_human_review` bucket remains in this triage.")
    lines.extend(
        [
            "",
            "## High-Risk Path-Level Detail",
            "",
            "The generated local `migration_map.json` contains the complete table. "
            "The current high-risk table is summarized here:",
            "",
        ]
    )
    for item in migration_map.triage_items:
        if item.why_high_risk:
            lines.append(
                f"- `{item.old_path}` -> `{item.bucket}`; "
                f"blocks_spec3=`{item.blocks_spec3}`; "
                f"destination=`{item.proposed_destination_path_or_area}`"
            )
    lines.append("")
    return "\n".join(lines)


def has_untriaged_high_risk(migration_map: MigrationMap) -> bool:
    return any(
        item.why_high_risk and item.bucket == "needs_human_review"
        for item in migration_map.triage_items
    )


def _classify_item(
    *,
    old_path: str,
    old_role: str,
    old_status: str,
    new_paths: tuple[str, ...],
) -> tuple[str, str, str, str, bool, str]:
    del old_status
    lower = old_path.lower()
    if new_paths:
        return (
            "duplicate_or_merged",
            "RADJAX-Tome",
            ", ".join(new_paths),
            "P3",
            False,
            "Audit found an existing new path; verify semantic equivalence.",
        )
    if _is_historical(lower):
        return (
            "historical_deprecated",
            "none",
            "historical archive",
            "P4",
            False,
            "Historical/deprecated naming or obsolete workflow.",
        )
    if _is_contract(lower, old_role):
        return (
            "belongs_contract",
            "RADJAX-Contract",
            _contract_area(lower),
            "P2",
            False,
            "Shared schema/contract concern; track outside Tome unless a "
            "wrapper is needed.",
        )
    if _is_student(lower, old_role):
        return (
            "belongs_student",
            "RADJAX-Student",
            _student_area(lower),
            "P3",
            False,
            "Student-side runtime/training/eval concern.",
        )
    if old_role == "mixed_producer_consumer" or _is_mixed(lower):
        return (
            "mixed_requires_split",
            "RADJAX-Tome and RADJAX-Student",
            _mixed_area(lower),
            "P1",
            True,
            "Split producer artifact creation from student consumption/training.",
        )
    if _is_must_before_spec3(lower, old_role):
        return (
            "must_migrate_tome_before_spec3",
            "RADJAX-Tome",
            _tome_area(lower),
            "P0",
            True,
            "Required before Contract-valid Tome emission can be implemented safely.",
        )
    if _is_before_full_burn(lower, old_role):
        return (
            "migrate_tome_before_full_burn",
            "RADJAX-Tome",
            _tome_area(lower),
            "P1",
            False,
            "Producer-side workflow needed before serious full-burn runs.",
        )
    if old_role in {"producer_doc", "producer_fixture", "unknown_needs_human_review"}:
        return (
            "migrate_tome_deferred",
            "RADJAX-Tome",
            "docs or fixtures",
            "P3",
            False,
            "Producer-related support material; defer unless pulled in by a "
            "migration chunk.",
        )
    return (
        "needs_human_review",
        "unknown",
        "unknown",
        "P0",
        True,
        "No deterministic triage rule matched.",
    )


def _linked_bucket_for_test(old_path: str) -> str:
    lower = old_path.lower()
    if _is_contract(lower, "producer_test"):
        return "belongs_contract"
    if _is_student(lower, "producer_test"):
        return "belongs_student"
    if _is_historical(lower):
        return "historical_deprecated"
    if any(
        token in lower
        for token in (
            "teacher_textbook",
            "teacher_target_store",
            "topk_tail",
            "cascaded_soft_labels_textbook",
            "export_teacher_targets",
            "inspect_targets",
            "target_manifest",
            "multishard_target_store",
            "offline_target_consumption",
            "tiny_dataset_pipeline",
            "prompt_corpus",
            "hf_teacher_backend",
            "teacher_export",
            "real_teacher_offline",
        )
    ):
        return "must_migrate_tome_before_spec3"
    if any(
        token in lower
        for token in (
            "fingerprint",
            "exemplar",
            "corridor",
            "real_teacher",
            "quality_per_byte",
            "mini_eval",
            "first_serious_burn",
        )
    ):
        return "mixed_requires_split"
    return "migrate_tome_deferred"


def _test_notes(old_path: str, linked_bucket: str) -> str:
    if linked_bucket == "must_migrate_tome_before_spec3":
        return "Port with Spec 2.7 or Spec 2.8 producer work."
    if linked_bucket == "mixed_requires_split":
        return "Port producer assertions after mixed file split, likely Spec 2.9."
    if linked_bucket == "belongs_student":
        return "Track in RADJAX-Student if still needed."
    if linked_bucket == "belongs_contract":
        return "Track in RADJAX-Contract if still needed."
    return "Defer unless the associated producer feature is migrated."


def _is_contract(path: str, old_role: str) -> bool:
    return (
        old_role == "contract_only"
        or "/contracts/" in path
        or "vocab_contract" in path
        or "student_artifact_contract" in path
        or "schema" in path
        or "compatibility" in path
    )


def _is_student(path: str, old_role: str) -> bool:
    if old_role == "mixed_producer_consumer":
        return False
    return any(
        token in path
        for token in (
            "student_backend",
            "student_config",
            "student_runtime",
            "student_logits",
            "train_student",
            "optimizer",
            "checkpoint",
            "run_distill",
            "kernel_runtime",
            "pjit",
            "pmap",
            "tpu_distill",
        )
    )


def _is_mixed(path: str) -> bool:
    return any(
        token in path
        for token in (
            "first_serious_burn",
            "full_distillation",
            "run_distill_stage",
            "student_artifact",
            "fingerprint/real_teacher",
            "fingerprint/exemplar_pass",
            "training/fingerprint",
        )
    )


def _is_must_before_spec3(path: str, old_role: str) -> bool:
    return old_role in {"producer_core", "producer_validation"} or any(
        token in path
        for token in (
            "teacher_textbook",
            "teacher_target",
            "target_store",
            "targets/multishard",
            "target_manifest",
            "export_teacher_targets",
            "inspect_targets",
            "validate_teacher",
            "validate_pipeline",
            "topk_tail",
            "cascaded_soft_labels",
            "teacher_export",
            "teachers/hf",
            "teachers/backend",
            "teachers/emission",
            "teachers/synthetic",
            "prompt_corpus",
            "split_prompt",
            "tokenize_corpus",
            "create_prompt_manifest",
            "tiny_dataset_pipeline",
        )
    )


def _is_before_full_burn(path: str, old_role: str) -> bool:
    return old_role == "producer_capture" or any(
        token in path
        for token in (
            "fingerprint",
            "exemplar",
            "corridor",
            "mini_eval",
            "real_teacher",
            "quality_per_byte",
            "export_smoke",
            "hf_teacher_specimen",
            "create_fake_targets",
            "generate_multiscale",
            "multiscale_shape",
            "plan_model_scale",
            "resolve_qwen_policy",
            "mode_plateau",
            "two_cycle",
            "validate_local",
        )
    )


def _is_historical(path: str) -> bool:
    return any(token in path for token in ("deprecated", "legacy_unused", "archive"))


def _contract_area(path: str) -> str:
    if "vocab" in path:
        return "vocab compatibility and contract schemas"
    if "manifest" in path:
        return "artifact manifest schema"
    return "shared artifact/schema validation"


def _student_area(path: str) -> str:
    if "optimizer" in path:
        return "optimizer/checkpoint training internals"
    if "student" in path:
        return "student backend/runtime"
    return "student training or evaluation workflow"


def _mixed_area(path: str) -> str:
    if "fingerprint" in path:
        return "split fingerprint producer artifacts from student training/eval"
    if "burn" in path or "distill" in path:
        return (
            "split burn/distillation orchestration into producer inputs vs "
            "student execution"
        )
    return "split producer artifact logic from consumer/runtime logic"


def _tome_area(path: str) -> str:
    if "script" in path or path.startswith("scripts/"):
        return "producer CLI surface"
    if "teacher_export" in path or "teachers/" in path:
        return "teacher backend/export producer surface"
    if "target" in path:
        return "teacher target store/export/validation surface"
    if "prompt" in path or "corpus" in path:
        return "corpus/source identity producer surface"
    if "fingerprint" in path:
        return "producer fingerprint artifact surface"
    return "RADJAX-Tome producer surface"


def _count_items_by_bucket(items) -> dict[str, int]:
    return dict(Counter(item.bucket for item in items))


def _open_questions(triage_items: tuple[TriageItem, ...]) -> tuple[str, ...]:
    questions = []
    if any(item.bucket == "belongs_contract" for item in triage_items):
        questions.append(
            "Which contract-bound schemas need RADJAX-Contract issues before "
            "Tome migration continues?"
        )
    if any(item.bucket == "mixed_requires_split" for item in triage_items):
        questions.append(
            "For mixed fingerprint/burn files, which producer outputs belong "
            "in Tome and which execution paths belong in Student?"
        )
    if any(item.bucket == "migrate_tome_deferred" for item in triage_items):
        questions.append(
            "Which deferred docs/fixtures are still useful as regression evidence?"
        )
    return tuple(questions)


def _summary_table(payload: dict[str, int]) -> str:
    if not payload:
        return "| Metric | Count |\n| --- | ---: |\n| none | 0 |"
    lines = ["| Metric | Count |", "| --- | ---: |"]
    for key in sorted(payload):
        lines.append(f"| {key} | {payload[key]} |")
    return "\n".join(lines)
