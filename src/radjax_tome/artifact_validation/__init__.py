"""Builder-independent validation primitives for materialized Tome artifacts.

This package owns reusable *artifact* validation only.  Production builders
may call it, but package materialization and inspection never need to import
``radjax_tome.builder`` to obtain validation semantics.
"""

from radjax_tome.artifact_validation.long_tail import long_tail_summary
from radjax_tome.artifact_validation.selection import (
    C6IntegrationError,
    load_curriculum_route_records,
    validate_integrated_selection_contract,
)

__all__ = [
    "C6IntegrationError",
    "load_curriculum_route_records",
    "long_tail_summary",
    "validate_integrated_selection_contract",
]
