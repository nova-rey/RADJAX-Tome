"""Textual implementation of the canonical M11 wizard surface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from radjax_tome.builder.config import canonical_production_build_intent
from radjax_tome.builder.config_io import tome_build_intent_document
from radjax_tome.tui.controller import (
    draft_text,
    load_draft,
    parse_draft_text,
    preflight_draft,
    save_draft_text_as,
    saved_document_matches,
)
from radjax_tome.tui.draft import WizardDraft
from radjax_tome.tui.process import run_canonical_cli, run_package_cli
from radjax_tome.tui.screens import SCREEN_NAMES
from radjax_tome.tui.screens_runtime import ConfirmScreen


def _blank_corpus_document() -> dict[str, Any]:
    return {
        "schema_version": "radjax_tome_corpus_build_intent_v1",
        "artifact": {"schema_version": "radjax_tome_corpus_artifact_v2"},
        "sources": [
            {
                "source_id": "source",
                "adapter": "local_text_tree_v1",
                "path": "sources",
            }
        ],
        "policy": {
            "normalization": "text_normalize_lf_strip_trailing_ws_v1",
            "filtering": {"min_chars": 1},
            "chunking": {"name": "char_window_v1", "max_chars": 1024},
            "deduplication": {"enabled": True},
            "ordering": "declared_source_ordinal_logical_locator_chunk_index_v1",
            "tokenizer": "smoke",
        },
        "layout": {"shard_capacity": 1024},
        "resources": {"memory_limit": "64MB", "worker_count": 1},
        "output": {"artifact_path": "artifact"},
        "execution": {"resume": False, "overwrite": False},
        "reporting": {"progress": True},
    }


def _blank_production_document() -> dict[str, Any]:
    intent = canonical_production_build_intent(
        teacher_model="teacher-model",
        dataset_path=Path("corpus"),
        corpus_manifest_path=Path("corpus"),
        teacher_model_provenance_path=Path("teacher-provenance.json"),
        output_dir=Path("production-artifact"),
        tokenizer_id="smoke",
    )
    return tome_build_intent_document(intent)


class WizardApp(App[None]):
    """Keyboard-first wizard delegating all semantics to canonical owners."""

    BINDINGS = [
        ("ctrl+s", "save_as", "Save As"),
        ("ctrl+q", "request_quit", "Quit"),
        ("escape", "request_quit", "Cancel"),
        ("ctrl+c", "request_cancel", "Interrupt"),
    ]
    CSS_PATH = "app.tcss"

    def __init__(self, workflow: str, config: Path | None = None) -> None:
        super().__init__()
        self.workflow = workflow
        self.draft = (
            load_draft(workflow, config)
            if config
            else WizardDraft(
                workflow,
                _blank_corpus_document()
                if workflow == "corpus"
                else _blank_production_document(),
            )
        )
        self._running = False
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_event: asyncio.Event | None = None
        self._force_event: asyncio.Event | None = None
        self._cancel_requested = False

    def _field_specs(self) -> list[tuple[str, str, str]]:
        document = self.draft.document
        if self.workflow == "corpus":
            source = document.get("sources", [{}])[0]
            return [
                ("Source path", "source-path", str(source.get("path", ""))),
                ("Source adapter", "source-adapter", str(source.get("adapter", ""))),
                (
                    "Logical record-ID field",
                    "record-id-field",
                    str(source.get("record_id_field") or ""),
                ),
                (
                    "Tokenizer binding",
                    "corpus-tokenizer",
                    str(document.get("policy", {}).get("tokenizer", "")),
                ),
                (
                    "Artifact destination",
                    "artifact-path",
                    str(document.get("output", {}).get("artifact_path", "")),
                ),
            ]
        corpus = document.get("corpus", {})
        teacher = document.get("teacher", {})
        outputs = document.get("outputs", {})
        return [
            ("Teacher/model", "teacher-model", str(teacher.get("model", ""))),
            ("Tokenizer ID", "tokenizer-id", str(teacher.get("tokenizer_id") or "")),
            (
                "Validated corpus artifact",
                "corpus-artifact",
                str(corpus.get("artifact_path", corpus.get("dataset_path", ""))),
            ),
            (
                "Expected corpus identity",
                "corpus-identity",
                str(corpus.get("expected_semantic_identity") or ""),
            ),
            (
                "Production destination",
                "output-dir",
                str(outputs.get("output_dir", "")),
            ),
            ("Runtime mode", "runtime-mode", str(teacher.get("runtime_mode", ""))),
        ]

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="wizard-scroll"):
            yield Label(f"{self.workflow.title()} wizard", id="title")
            yield Static(" → ".join(SCREEN_NAMES), id="screens")
            yield Static(
                "Canonical configuration only. Editing and preflight do not load "
                "models, allocate accelerators, stage artifacts, or package output.",
                id="notice",
            )
            with TabbedContent(initial="start-pane", id="wizard-tabs"):
                with TabPane("Start", id="start-pane"):
                    yield Static(
                        "Choose a canonical workflow, then complete Inputs, "
                        "Behavior, Resources/Destination, and Review."
                    )
                with TabPane("Inputs", id="inputs-pane"):
                    yield Label("Workflow-specific inputs", classes="section-label")
                    for label, field_id, value in self._field_specs()[:4]:
                        yield Label(label, classes="field-label")
                        yield Input(value=value, id=field_id)
                with TabPane("Behavior", id="behavior-pane"):
                    yield Label(
                        "Canonical policy and advanced behavior remain in the "
                        "JSON editor; no hidden defaults are added.",
                        classes="section-label",
                    )
                    if self.workflow == "corpus":
                        yield Label(
                            "Normalization and ordering are canonical policy IDs."
                        )
                        yield Label(
                            "Exact deduplication is controlled by the canonical policy."
                        )
                    else:
                        yield Label(
                            "Production preset and selection fields remain reloadable."
                        )
                        yield Label("GPU/device choices are not TUI-only semantics.")
                with TabPane("Resources / Destination", id="resources-pane"):
                    yield Label("Destination and runtime", classes="section-label")
                    for label, field_id, value in self._field_specs()[4:]:
                        yield Label(label, classes="field-label")
                        yield Input(value=value, id=field_id)
                with TabPane("Review / Advanced", id="review-pane"):
                    yield Label("Advanced canonical JSON", classes="section-label")
                    yield TextArea(
                        draft_text(self.draft),
                        language="json",
                        id="config-editor",
                        placeholder="Edit every supported canonical field here",
                        show_line_numbers=True,
                    )
            yield Static("Ready", id="status")
            with Horizontal(id="actions"):
                yield Button("Apply fields", id="apply-fields")
                yield Button("Preflight", id="preflight")
                yield Button("Run", id="run")
                yield Button("Save As", id="save")
                yield Button("Package", id="package")
                yield Button("Quit", id="quit")
            yield Input(
                placeholder="New config path for Save As",
                id="config-path",
            )
            yield Input(
                placeholder="Package destination (post-build only)",
                id="package-output",
            )
            yield Input(
                value="student", placeholder="Package profile", id="package-profile"
            )
            yield Input(
                value="directory",
                placeholder="Package transport",
                id="package-transport",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#config-path", Input).focus()

    def on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == "ctrl+s":
            event.stop()
            self.action_save_as()
        elif key == "ctrl+c":
            event.stop()
            self.action_request_cancel()

    def on_resize(self, event: object) -> None:
        size = getattr(event, "size", None)
        if size is not None and (size.width < 80 or size.height < 24):
            self.query_one("#notice", Static).update(
                "Small terminal: resize toward 80x24 for the full wizard; canonical "
                "fields remain available by scrolling."
            )

    def action_request_quit(self) -> None:
        if self._running:
            self.action_request_cancel()
            return
        self.push_screen(ConfirmScreen(), self._finish_quit)

    def action_request_cancel(self) -> None:
        if not self._running:
            self._set_status("No canonical workflow is running.")
            return
        if self._cancel_requested:
            self.push_screen(
                ConfirmScreen("The child is still running. Force-stop it?"),
                self._finish_force_stop,
            )
            return
        self.push_screen(
            ConfirmScreen("Interrupt the canonical workflow and preserve its state?"),
            self._finish_cancel,
        )

    def _finish_quit(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()

    def _finish_cancel(self, confirmed: bool | None) -> None:
        if confirmed:
            self._cancel_requested = True
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._set_status("Interrupt requested; preserving workspace and staging.")

    def _finish_force_stop(self, confirmed: bool | None) -> None:
        if confirmed and self._force_event is not None:
            self._force_event.set()
            self._set_status(
                "Force-stop requested; workspace and staging are preserved."
            )

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def action_save_as(self) -> None:
        self._save()

    def _apply_form_fields(self) -> dict[str, Any]:
        source_path = self.draft.saved_path or Path("wizard.json")
        document = parse_draft_text(
            self.workflow,
            self.query_one("#config-editor", TextArea).text,
            source_path=source_path,
        )
        values = {
            field_id: self.query_one(f"#{field_id}", Input).value.strip()
            for _, field_id, _ in self._field_specs()
        }
        if self.workflow == "corpus":
            source = document["sources"][0]
            for key, field_id in (
                ("path", "source-path"),
                ("adapter", "source-adapter"),
                ("record_id_field", "record-id-field"),
            ):
                if values[field_id]:
                    source[key] = values[field_id]
                elif key == "record_id_field":
                    source.pop(key, None)
            if values["corpus-tokenizer"]:
                document["policy"]["tokenizer"] = values["corpus-tokenizer"]
            if values["artifact-path"]:
                document["output"]["artifact_path"] = values["artifact-path"]
        else:
            document["teacher"]["model"] = values["teacher-model"]
            document["teacher"]["tokenizer_id"] = values["tokenizer-id"] or None
            document["teacher"]["runtime_mode"] = values["runtime-mode"]
            corpus = document["corpus"]
            artifact = values["corpus-artifact"]
            if "artifact_path" in corpus:
                corpus["artifact_path"] = artifact
            else:
                corpus["dataset_path"] = artifact
                corpus["corpus_manifest_path"] = artifact
            if "expected_semantic_identity" in corpus:
                corpus["expected_semantic_identity"] = values["corpus-identity"] or None
            document["outputs"]["output_dir"] = values["output-dir"]
        self.draft.document = document
        self.query_one("#config-editor", TextArea).text = draft_text(self.draft)
        return document

    def _saved_editor_bytes_match(self) -> bool:
        try:
            self._apply_form_fields()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_status(f"Configuration is invalid: {exc}")
            return False
        text = self.query_one("#config-editor", TextArea).text
        if not saved_document_matches(self.draft, text):
            self._set_status(
                "Unsaved canonical changes detected; use Save As before "
                "preflight or run."
            )
            return False
        return True

    def _save(self) -> None:
        self._apply_form_fields()
        destination = self.query_one("#config-path", Input).value.strip()
        if not destination:
            self._set_status("Enter a new config filename before Save As.")
            return
        try:
            saved = save_draft_text_as(
                self.draft,
                destination,
                self.query_one("#config-editor", TextArea).text,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_status(f"Save failed: {exc}")
            return
        self._set_status(f"Saved canonical configuration: {saved}")

    def _preflight(self) -> None:
        if not self._saved_editor_bytes_match():
            return
        try:
            report = preflight_draft(self.draft)
        except (OSError, TypeError, ValueError) as exc:
            self._set_status(f"Preflight failed: {exc}")
            return
        self._set_status(f"Preflight passed: {report.get('action', 'ready')}")

    def _request_run(self) -> None:
        if self._running:
            self._set_status("A canonical workflow is already running.")
            return
        self.push_screen(
            ConfirmScreen("Run the canonical workflow now?"),
            self._finish_run_confirmation,
        )

    def _finish_run_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(self._run(), exclusive=True)

    def _set_process(self, process: asyncio.subprocess.Process | None) -> None:
        self._process = process

    async def _run(self) -> None:
        if self.draft.saved_path is None or not self._saved_editor_bytes_match():
            return
        self._running = True
        self._cancel_requested = False
        self._cancel_event = asyncio.Event()
        self._force_event = asyncio.Event()
        try:
            preflight_draft(self.draft)
            self._set_status("Preflight passed; running canonical CLI...")
            result = await run_canonical_cli(
                self.workflow,
                self.draft.saved_path,
                cancel_event=self._cancel_event,
                force_event=self._force_event,
                process_holder=self._set_process,
            )
            if result.returncode == 130:
                self._set_status(
                    "Canonical workflow interrupted; workspace and staging preserved."
                )
            elif result.result is None:
                self._set_status(f"CLI transport failed: {result.transport_error}")
            elif result.returncode:
                self._set_status(f"CLI failed ({result.returncode}); see stderr.")
            else:
                self._set_status("Canonical workflow completed successfully.")
        except (OSError, TypeError, ValueError) as exc:
            self._set_status(f"Run failed: {exc}")
        finally:
            self._running = False
            self._process = None
            self._cancel_event = None
            self._force_event = None

    def _request_package(self) -> None:
        if self.workflow != "production":
            self._set_status("Packaging is a post-build action for production only.")
            return
        self.push_screen(
            ConfirmScreen("Package the completed workspace now?"),
            self._finish_package_confirmation,
        )

    def _finish_package_confirmation(self, confirmed: bool | None) -> None:
        if confirmed:
            self.run_worker(self._package(), exclusive=True)

    async def _package(self) -> None:
        if self.draft.saved_path is None:
            self._set_status("Save a production configuration before packaging.")
            return
        output = self.query_one("#package-output", Input).value.strip()
        if not output:
            self._set_status("Enter a package destination first.")
            return
        try:
            intent = load_draft(self.workflow, self.draft.saved_path)
            workspace = Path(intent.document["outputs"]["output_dir"])
            result = await run_package_cli(
                workspace,
                output,
                profile=self.query_one("#package-profile", Input).value.strip(),
                transport=self.query_one("#package-transport", Input).value.strip(),
            )
            self._set_status(
                "Package completed successfully."
                if result.returncode == 0
                else f"Package failed ({result.returncode}); see stderr."
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            self._set_status(f"Package failed: {exc}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "apply-fields": lambda: self._apply_form_fields(),
            "quit": self.action_request_quit,
            "save": self._save,
            "preflight": self._preflight,
            "run": self._request_run,
            "package": self._request_package,
        }
        action = actions.get(event.button.id)
        if action is not None:
            action()


__all__ = ["WizardApp"]
