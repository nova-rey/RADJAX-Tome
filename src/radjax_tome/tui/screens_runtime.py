"""Runtime-only confirmation screen; imported only when Textual is available."""

from textual.screen import ModalScreen
from textual.widgets import Button, Label


class ConfirmScreen(ModalScreen[bool]):
    def __init__(self, message: str = "Exit without starting a build?") -> None:
        super().__init__()
        self.message = message

    def compose(self):
        yield Label(self.message, id="confirm-label")
        yield Button("Exit", id="confirm-exit")
        yield Button("Stay", id="confirm-stay")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-exit")


__all__ = ["ConfirmScreen"]
