from __future__ import annotations

from typing import Any

from radjax_tome._lazy_exports import (
    LazyExportMap,
    lazy_export_names,
    resolve_lazy_export,
)
from radjax_tome.audit.selected_linkage import (
    AUDIT_SCHEMA_VERSION,
    SelectedLinkageAuditReport,
    audit_selected_linkage,
    write_selected_linkage_audit,
)

_LAZY_EXPORTS: LazyExportMap = {
    "RefactorAudit": ("radjax_tome.audit.refactor_surface", "RefactorAudit"),
    "run_refactor_audit": (
        "radjax_tome.audit.refactor_surface",
        "run_refactor_audit",
    ),
    "write_refactor_audit": (
        "radjax_tome.audit.refactor_surface",
        "write_refactor_audit",
    ),
}

__all__ = [
    "RefactorAudit",
    "AUDIT_SCHEMA_VERSION",
    "SelectedLinkageAuditReport",
    "audit_selected_linkage",
    "run_refactor_audit",
    "write_selected_linkage_audit",
    "write_refactor_audit",
]


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), _LAZY_EXPORTS, name)


def __dir__() -> list[str]:
    return lazy_export_names(globals(), _LAZY_EXPORTS)
