"""Small Textual controller over canonical wizard documents."""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Static

from radjax_tome.tui.controller import load_draft
from radjax_tome.tui.draft import WizardDraft
from radjax_tome.tui.screens import SCREEN_NAMES


class WizardApp(App[None]):
    """Keyboard-first single-column wizard shell."""

    BINDINGS = [
        ("ctrl+s", "save_as", "Save As"),
        ("ctrl+q", "request_quit", "Quit"),
    ]
    CSS_PATH = "app.tcss"

    def __init__(self, workflow: str, config: Path | None = None) -> None:
        super().__init__()
        self.workflow = workflow
        self.draft: WizardDraft | None = (
            load_draft(workflow, config) if config else None
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="wizard-scroll"):
            yield Label(f"{self.workflow.title()} wizard", id="title")
            yield Static(" → ".join(SCREEN_NAMES), id="screens")
            yield Static(
                "Canonical configuration only. No model, accelerator, staging, or "
                "output mutation occurs while editing.",
                id="notice",
            )
            yield Input(
                value=(
                    str(self.draft.saved_path)
                    if self.draft and self.draft.saved_path
                    else ""
                ),
                placeholder="Config path (optional)",
                id="config-path",
            )
            yield Static("Ready", id="status")
            yield Button("Preflight", id="preflight")
            yield Button("Run canonical workflow", id="run")
            yield Button("Save As", id="save")
            yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#config-path", Input).focus()

    def action_request_quit(self) -> None:
        self.push_screen(
            __import__(
                "radjax_tome.tui.screens_runtime", fromlist=["ConfirmScreen"]
            ).ConfirmScreen()
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.action_request_quit()
        elif event.button.id == "preflight":
            self.query_one("#status", Static).update(
                "Preflight is delegated to the canonical CLI."
            )
        elif event.button.id == "run":
            self.query_one("#status", Static).update(
                "Run requires a saved config and explicit confirmation."
            )
        elif event.button.id == "save":
            self.query_one("#status", Static).update(
                "Save As requires a new filename; existing files are never replaced."
            )


__all__ = ["WizardApp"]
