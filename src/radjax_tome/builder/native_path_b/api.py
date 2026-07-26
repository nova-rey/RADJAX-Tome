"""Typed, import-isolated boundary for the native C6 Path-B request.

This module intentionally knows only the exact configuration gate.  It must
not import the production facade or any runtime algorithm: the facade supplies
the compatibility executor during M3C, while M4 will replace that executor
with extracted native stages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

CANONICAL_TARGET_POLICY = "corridor_exemplar_v1"
CANONICAL_SELECTION_INTEGRATION_POLICY = "corridor_first_global_backfill_v1"
CANONICAL_DELIVERY_PATH = "two_pass_rerun_selected"


class _PathBConfigSource(Protocol):
    """The deliberately small ProductionBuildConfig surface needed for routing."""

    target_policy: str
    selection_integration_policy: str
    exemplar_selection_enabled: bool
    exemplar_delivery_path: str | None
    total_selected_exemplar_budget: int | None


@dataclass(frozen=True)
class CanonicalPathBConfig:
    """Resolved native request with its original compatibility configuration."""

    source_config: _PathBConfigSource
    target_policy: str
    selection_integration_policy: str
    exemplar_delivery_path: str
    total_selected_exemplar_budget: int


def resolve_canonical_path_b_config(
    config: _PathBConfigSource,
) -> CanonicalPathBConfig | None:
    """Return a native request only for the exact supported C6 Path-B tuple.

    The caller retains validation and compatibility behavior for every other
    configuration.  In particular, aliases are not normalized here.
    """

    if config.target_policy != CANONICAL_TARGET_POLICY:
        return None
    if config.selection_integration_policy != CANONICAL_SELECTION_INTEGRATION_POLICY:
        return None
    if config.exemplar_selection_enabled is not True:
        return None
    if config.exemplar_delivery_path != CANONICAL_DELIVERY_PATH:
        return None
    if config.total_selected_exemplar_budget is None:
        return None
    return CanonicalPathBConfig(
        source_config=config,
        target_policy=CANONICAL_TARGET_POLICY,
        selection_integration_policy=CANONICAL_SELECTION_INTEGRATION_POLICY,
        exemplar_delivery_path=CANONICAL_DELIVERY_PATH,
        total_selected_exemplar_budget=config.total_selected_exemplar_budget,
    )


ResultT = TypeVar("ResultT")


def run_canonical_path_b(
    config: CanonicalPathBConfig,
    *,
    compatibility_executor: Callable[[_PathBConfigSource], ResultT],
) -> ResultT:
    """Delegate without importing production or runtime algorithms.

    M3C preserves current behavior by executing the source configuration.  The
    executor is injected by the production facade and is the only execution
    dependency this boundary accepts.
    """

    return compatibility_executor(config.source_config)
