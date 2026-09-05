"""Shared screen labels and keyboard bindings for the optional wizard."""

SCREEN_NAMES = (
    "Start",
    "Inputs",
    "Behavior",
    "Resources/Destination",
    "Review",
    "Preflight",
    "Save/Confirmation",
    "Run Status/Results",
)

KEY_BINDINGS = ("tab", "shift+tab", "enter", "escape", "ctrl+s", "ctrl+q", "ctrl+c")

__all__ = ["KEY_BINDINGS", "SCREEN_NAMES"]
