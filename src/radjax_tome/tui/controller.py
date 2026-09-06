"""Canonical owner bridge used by both wizard workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radjax_tome.builder.config import (
    production_build_config_from_resolved,
    resolve_tome_build_intent,
)
from radjax_tome.builder.config_io import (
    load_tome_build_intent,
    parse_tome_build_intent_document,
    tome_build_intent_document,
)
from radjax_tome.builder.production_stages.preflight import (
    assess_production_preflight,
    validate_required_inputs,
)
from radjax_tome.corpora.config import (
    corpus_build_intent_document,
    load_corpus_build_intent,
    parse_corpus_build_intent_document,
)
from radjax_tome.corpora.feasibility import assess_corpus_feasibility
from radjax_tome.io.config_export import save_as_config
from radjax_tome.tui.draft import WizardDraft


def load_draft(workflow: str, path: str | Path) -> WizardDraft:
    source = Path(path).resolve()
    if workflow == "corpus":
        document = corpus_build_intent_document(load_corpus_build_intent(source))
    else:
        document = tome_build_intent_document(load_tome_build_intent(source))
    return WizardDraft(
        workflow,
        document,
        source_path=source,
        saved_path=source,
        saved_bytes=source.read_bytes(),
    )


def draft_text(draft: WizardDraft) -> str:
    return (
        json.dumps(draft.document, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )


def save_draft_as(draft: WizardDraft, destination: str | Path) -> Path:
    if draft.workflow == "corpus":

        def parser(document: Any) -> Any:
            return parse_corpus_build_intent_document(document, source_path=destination)
    else:

        def parser(document: Any) -> Any:
            return parse_tome_build_intent_document(document, source_path=destination)

    target = save_as_config(
        destination,
        draft_text(draft),
        parse=lambda text: parser(json.loads(text)),
    )
    draft.saved_path = target
    draft.saved_bytes = target.read_bytes()
    return target


def parse_draft_text(
    workflow: str, text: str, *, source_path: str | Path
) -> dict[str, Any]:
    document = json.loads(text)
    if workflow == "corpus":
        return corpus_build_intent_document(
            parse_corpus_build_intent_document(document, source_path=source_path)
        )
    return tome_build_intent_document(
        parse_tome_build_intent_document(document, source_path=source_path)
    )


def save_draft_text_as(draft: WizardDraft, destination: str | Path, text: str) -> Path:
    target = Path(destination).resolve()
    canonical = parse_draft_text(draft.workflow, text, source_path=target)
    saved = save_as_config(
        target,
        json.dumps(canonical, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        parse=lambda value: parse_draft_text(draft.workflow, value, source_path=target),
    )
    draft.document = canonical
    draft.saved_path = saved
    draft.saved_bytes = saved.read_bytes()
    return saved


def saved_document_matches(draft: WizardDraft, text: str) -> bool:
    """Require the exact canonical bytes saved by Save As before execution."""

    if draft.saved_path is None:
        return False
    try:
        return draft.saved_path.read_bytes() == text.encode("utf-8")
    except OSError:
        return False


def preflight_draft(draft: WizardDraft) -> dict[str, Any]:
    if draft.workflow == "corpus":
        if draft.saved_path is None:
            raise ValueError("save the draft before preflight")
        return assess_corpus_feasibility(load_corpus_build_intent(draft.saved_path))
    if draft.saved_path is None:
        raise ValueError("save the draft before preflight")
    intent = load_tome_build_intent(draft.saved_path)
    resolved = resolve_tome_build_intent(intent, source="m11_tui")
    production = production_build_config_from_resolved(resolved)
    blockers: list[str] = []
    validate_required_inputs(production, blockers)
    if blockers:
        raise ValueError("; ".join(blockers))
    resume_resolution: dict[str, Any] | None = None
    if intent.execution.resume:
        from radjax_tome.builder.native_path_b.api import (
            resolve_canonical_path_b_config,
        )
        from radjax_tome.builder.native_path_b.resume import (
            resolve_native_path_b_resume,
        )

        resolution = resolve_native_path_b_resume(
            intent.outputs.output_dir,
            config=resolve_canonical_path_b_config(production),
        )
        resume_resolution = {
            "complete": resolution.complete,
            "stage": resolution.stage,
            "reason": resolution.failure.reason if resolution.failure else None,
        }
    assessment = assess_production_preflight(
        intent.outputs.output_dir,
        config=production,
        resume=intent.execution.resume,
        overwrite=intent.execution.overwrite,
    )
    if assessment.status != "pass":
        raise ValueError("; ".join(assessment.blockers))
    return {
        "status": "pass",
        "workflow": "production",
        "action": assessment.action,
        "selection_authority_hash": resolved.selection_authority_hash,
        "resume_resolution": resume_resolution,
        "transformed": False,
    }


__all__ = [
    "WizardDraft",
    "draft_text",
    "load_draft",
    "preflight_draft",
    "parse_draft_text",
    "save_draft_as",
    "save_draft_text_as",
    "saved_document_matches",
]
