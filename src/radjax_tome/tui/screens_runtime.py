"""Runtime-only confirmation screen; imported only when Textual is available."""

from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmScreen(ModalScreen[bool]):
    def compose(self):
        yield Label("Exit without starting a build?", id="confirm-label")
        yield Button("Exit", id="confirm-exit")
        yield Button("Stay", id="confirm-stay")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-exit")


__all__ = ["ConfirmScreen"]
