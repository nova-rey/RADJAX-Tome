"""Textual implementation of the canonical M11 wizard surface."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Static, TextArea

from radjax_tome.tui.controller import (
    draft_text,
    load_draft,
    preflight_draft,
    save_draft_text_as,
)
from radjax_tome.tui.draft import WizardDraft
from radjax_tome.tui.process import run_canonical_cli, run_package_cli
from radjax_tome.tui.screens import SCREEN_NAMES
from radjax_tome.tui.screens_runtime import ConfirmScreen


class WizardApp(App[None]):
    """Keyboard-first wizard delegating all semantics to canonical owners."""

    BINDINGS = [
        ("ctrl+s", "save_as", "Save As"),
        ("ctrl+q", "request_quit", "Quit"),
        ("escape", "request_quit", "Cancel"),
    ]
    CSS_PATH = "app.tcss"

    def __init__(self, workflow: str, config: Path | None = None) -> None:
        super().__init__()
        self.workflow = workflow
        self.draft: WizardDraft | None = (
            load_draft(workflow, config) if config else None
        )
        self._running = False

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
            yield Input(
                value=(
                    str(self.draft.saved_path)
                    if self.draft and self.draft.saved_path
                    else ""
                ),
                placeholder="New config path for Save As",
                id="config-path",
            )
            yield TextArea(
                draft_text(self.draft) if self.draft else "",
                language="json",
                id="config-editor",
                placeholder="Paste a complete canonical configuration here",
                show_line_numbers=True,
            )
            yield Static("Ready", id="status")
            with Horizontal(id="actions"):
                yield Button("Preflight", id="preflight")
                yield Button("Run", id="run")
                yield Button("Save As", id="save")
                yield Button("Package", id="package")
                yield Button("Quit", id="quit")
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

    def on_resize(self, event: object) -> None:
        size = getattr(event, "size", None)
        if size is not None and (size.width < 80 or size.height < 24):
            self.query_one("#notice", Static).update(
                "Small terminal: resize toward 80x24 for the full wizard; canonical "
                "fields remain available by scrolling."
            )

    def action_request_quit(self) -> None:
        self.push_screen(ConfirmScreen(), self._finish_quit)

    def _finish_quit(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit()

    def action_save_as(self) -> None:
        self._save()

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _save(self) -> None:
        destination = self.query_one("#config-path", Input).value.strip()
        if not destination:
            self._set_status("Enter a new config filename before Save As.")
            return
        try:
            if self.draft is None:
                self.draft = WizardDraft(self.workflow, {})
            saved = save_draft_text_as(
                self.draft,
                destination,
                self.query_one("#config-editor", TextArea).text,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._set_status(f"Save failed: {exc}")
            return
        self._set_status(f"Saved canonical configuration: {saved}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "quit": self.action_request_quit,
            "save": self._save,
            "preflight": self._preflight,
            "run": self._request_run,
            "package": self._request_package,
        }
        action = actions.get(event.button.id)
        if action is not None:
            action()

    def _preflight(self) -> None:
        if self.draft is None or self.draft.saved_path is None:
            self._set_status("Save As a canonical configuration before preflight.")
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

    async def _run(self) -> None:
        if self.draft is None or self.draft.saved_path is None:
            self._set_status("Save As a canonical configuration before running.")
            return
        self._running = True
        try:
            preflight_draft(self.draft)
            self._set_status("Preflight passed; running canonical CLI...")
            result = await run_canonical_cli(self.workflow, self.draft.saved_path)
            if result.result is None:
                self._set_status(f"CLI transport failed: {result.transport_error}")
            elif result.returncode:
                self._set_status(f"CLI failed ({result.returncode}); see stderr.")
            else:
                self._set_status("Canonical workflow completed successfully.")
        except (OSError, TypeError, ValueError) as exc:
            self._set_status(f"Run failed: {exc}")
        finally:
            self._running = False

    def _request_package(self) -> None:
        if (
            self.workflow != "production"
            or self.draft is None
            or self.draft.saved_path is None
        ):
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


__all__ = ["WizardApp"]
