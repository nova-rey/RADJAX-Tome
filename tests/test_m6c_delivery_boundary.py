"""M6C tests for stable delivery contracts during module extraction."""

from __future__ import annotations

from radjax_tome.builder.exemplar_delivery import (
    ExemplarDeliveryConfig as FacadeConfig,
)
from radjax_tome.builder.exemplar_delivery import (
    PreparedSelectedDelivery as FacadePrepared,
)
from radjax_tome.builder.exemplar_delivery_contracts import (
    ExemplarDeliveryConfig,
    PreparedSelectedDelivery,
)


def test_delivery_public_handoffs_are_forwarded_without_type_fork() -> None:
    assert FacadeConfig is ExemplarDeliveryConfig
    assert FacadePrepared is PreparedSelectedDelivery
