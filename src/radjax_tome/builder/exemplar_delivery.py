"""Compatibility façade for selected-exemplar delivery.

The implementation is partitioned under :mod:`radjax_tome.builder.delivery`.
This module preserves the historic import surface while production code uses
the focused owner modules.
"""

from __future__ import annotations

from radjax_tome.builder.delivery import _legacy as _legacy
from radjax_tome.builder.delivery._legacy import *  # noqa: F403

# Private helpers were historically importable despite not being public API.
# Keep them reachable through the compatibility façade while focused modules
# become the supported owner imports.  Tests of the implementation should
# target those owner modules rather than relying on this compatibility detail.
globals().update(
    {name: value for name, value in vars(_legacy).items() if not name.startswith("__")}
)

# Rebind public operations to their focused owners after compatibility symbols
# have been restored above.
from radjax_tome.builder.delivery.assembly import (  # noqa: E402
    assemble_selected_delivery_artifacts,
    finalize_selected_delivery_corridor,
)
from radjax_tome.builder.delivery.rerun import run_selected_delivery_rerun  # noqa: E402


def materialize_selected_exemplar_delivery(config):
    """Materialize through the canonical rerun, late corridor, assembly order."""
    prepared = run_selected_delivery_rerun(config)
    finalized = finalize_selected_delivery_corridor(prepared)
    return assemble_selected_delivery_artifacts(finalized)
