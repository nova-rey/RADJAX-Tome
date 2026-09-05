"""Canonical owner bridge used by both wizard workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from radjax_tome.builder.config_io import (
    load_tome_build_intent,
    parse_tome_build_intent_document,
    tome_build_intent_document,
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
    return WizardDraft(workflow, document, source_path=source, saved_path=source)


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
    return target


def preflight_draft(draft: WizardDraft) -> dict[str, Any]:
    if draft.workflow == "corpus":
        if draft.saved_path is None:
            raise ValueError("save the draft before preflight")
        return assess_corpus_feasibility(load_corpus_build_intent(draft.saved_path))
    if draft.saved_path is None:
        raise ValueError("save the draft before preflight")
    load_tome_build_intent(draft.saved_path)
    return {"status": "pass", "workflow": "production", "transformed": False}


__all__ = [
    "WizardDraft",
    "draft_text",
    "load_draft",
    "preflight_draft",
    "save_draft_as",
]
