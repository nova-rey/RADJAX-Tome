"""Stable machine-readable public CLI result models."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CLIWarning:
    code: str
    message: str
    phase: str = "command"


@dataclass(frozen=True)
class CLIError:
    code: str
    message: str
    phase: str
    location: str | None = None
    repair: str | None = None


@dataclass
class CLIResult:
    command: str
    status: str
    exit_code: int
    artifact: dict[str, Any] | None = None
    reports: dict[str, Any] = field(default_factory=dict)
    warnings: list[CLIWarning] = field(default_factory=list)
    error: CLIError | None = None
    config: dict[str, Any] | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    receipt_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = "radjax_tome_cli_result_v1"
        value["versions"] = {
            "radjax_tome": "0.1.0",
            "radjax_contract": "0.9.0",
            "radjax_contract_commit": (
                "373e3d17060d4ce1c4a0db6065c9289da714bde7"
            ),
        }
        return value
