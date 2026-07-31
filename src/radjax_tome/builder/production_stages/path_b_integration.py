"""Bind existing production callbacks to the sole native Path-B orchestrator.

This module owns callback composition only.  Stage semantics, persistence, and
ordering remain in :mod:`radjax_tome.builder.native_path_b`; callback bodies
remain independently movable production services.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from radjax_tome.builder.native_path_b.orchestrator import (
    SliceFiveOperations,
    SliceFourOperations,
    SliceThreeOperations,
    SliceTwoOperations,
    run_slice_five,
    run_slice_four,
    run_slice_three,
    run_slice_two,
)


@dataclass(frozen=True)
class NativePathBCallbacks:
    """Existing domain callbacks and terminal-report adapters for Path-B."""

    early_corridor: Callable[[], Any]
    fingerprint_authority: Callable[[], Any]
    global_authority: Callable[[Any], Any]
    integrated_selection: Callable[[Any], Any]
    selected_rerun: Callable[[Any], Any]
    late_corridor: Callable[[Any], Any]
    assembly: Callable[[Any], Any]
    validation_linkage: Callable[[Any], Any]
    reconciliation_cover: Callable[[Any], Any]
    final_reporting: Callable[[Any], Any]
    stage_failure: Callable[[Any], Any]
    selection_underfilled: Callable[[Any], Any]
    terminal_report: Callable[[], Any | None]


def run_post_score_path_b(
    canonical_config: Any,
    slice_one: Any,
    *,
    callbacks: NativePathBCallbacks,
) -> Any:
    """Run post-score Path-B in M4 order through injected existing callbacks."""

    slice_two = run_slice_two(
        canonical_config,
        slice_one,
        operations=SliceTwoOperations(
            early_corridor=lambda _, __: callbacks.early_corridor(),
            fingerprint_authority=lambda _, __: callbacks.fingerprint_authority(),
            global_authority=lambda _, __, fingerprint: callbacks.global_authority(
                fingerprint
            ),
        ),
    )
    if slice_two.status != "pass":
        return callbacks.stage_failure(
            slice_two.global_authority.failure
            if slice_two.global_authority is not None
            else (
                slice_two.fingerprint_authority.failure
                if slice_two.fingerprint_authority is not None
                else slice_two.early_corridor.failure
            )
        )

    slice_three = run_slice_three(
        canonical_config,
        slice_two,
        operations=SliceThreeOperations(
            integrated_selection=lambda _, authorities: callbacks.integrated_selection(
                authorities
            ),
        ),
    )
    if slice_three.status != "pass":
        failure = (
            None
            if slice_three.integrated_selection is None
            else slice_three.integrated_selection.failure
        )
        if failure is not None and any(
            blocker.startswith("C6 selected budget underfilled before selected rerun")
            for blocker in failure.blockers
        ):
            return callbacks.selection_underfilled(failure)
        return callbacks.stage_failure(failure)

    slice_four = run_slice_four(
        canonical_config,
        slice_three,
        operations=SliceFourOperations(
            selected_rerun=lambda _, inputs: callbacks.selected_rerun(inputs),
            late_corridor=lambda _, inputs: callbacks.late_corridor(inputs),
            assembly=lambda _, inputs: callbacks.assembly(inputs),
        ),
    )
    if slice_four.status != "pass":
        return callbacks.stage_failure(
            slice_four.assembly.failure
            if slice_four.assembly is not None
            else (
                slice_four.late_corridor.failure
                if slice_four.late_corridor is not None
                else (
                    slice_four.selected_rerun.failure
                    if slice_four.selected_rerun is not None
                    else None
                )
            )
        )

    slice_five = run_slice_five(
        canonical_config,
        slice_four,
        operations=SliceFiveOperations(
            validation_linkage=lambda _, inputs: callbacks.validation_linkage(inputs),
            reconciliation_cover=lambda _, inputs: callbacks.reconciliation_cover(
                inputs
            ),
            final_reporting=lambda _, inputs: callbacks.final_reporting(inputs),
        ),
    )
    terminal_report = callbacks.terminal_report()
    if slice_five.final_result is not None and terminal_report is not None:
        return terminal_report
    # Preserve the terminal stage's concrete failure.  Reporting, reconciliation,
    # and validation are all typed independently; collapsing a later failure onto
    # validation hid the real remediation behind a generic adapter error.
    failure = (
        slice_five.final_result.failure
        if slice_five.final_result is not None
        else (
            slice_five.reconciliation.failure
            if slice_five.reconciliation is not None
            else (
                slice_five.validation.failure
                if slice_five.validation is not None
                else None
            )
        )
    )
    return callbacks.stage_failure(failure)
