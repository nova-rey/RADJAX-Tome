"""Transitive import policy for the M6 production ownership boundary.

The policy is intentionally static: an intermediate forwarding module must not
be able to disguise an outward dependency from the package/import checks.
"""

from __future__ import annotations

import ast
from collections import deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "radjax_tome"
PACKAGE = "radjax_tome"


def _module_name(path: Path, *, root: Path = PACKAGE_ROOT) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _resolve_from(module: str, level: int, imported: str | None) -> str:
    if level == 0:
        return imported or ""
    package = module.rsplit(".", 1)[0]
    pieces = package.split(".")
    if level:
        pieces = pieces[: len(pieces) - level + 1]
    return ".".join((*pieces, *(imported.split(".") if imported else ())))


def _internal_imports(path: Path, modules: set[str], *, root: Path) -> set[str]:
    module = _module_name(path, root=root)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(module, node.level, node.module)
            candidates.append(base)
            candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        else:
            continue
        for candidate in candidates:
            if candidate in modules:
                found.add(candidate)
                continue
            # Imports of a package initializer reach the initializer; imports
            # of a symbol stay at their owning module.
            while "." in candidate:
                candidate = candidate.rsplit(".", 1)[0]
                if candidate in modules:
                    found.add(candidate)
                    break
    return found


def _module_graph(root: Path = PACKAGE_ROOT) -> dict[str, set[str]]:
    paths = tuple(root.rglob("*.py"))
    modules = {_module_name(path, root=root) for path in paths}
    return {
        _module_name(path, root=root): _internal_imports(path, modules, root=root)
        for path in paths
    }


def _find_path(
    graph: dict[str, set[str]], start: str, forbidden_prefix: str
) -> tuple[str, ...] | None:
    pending: deque[tuple[str, tuple[str, ...]]] = deque([(start, (start,))])
    visited = {start}
    while pending:
        node, path = pending.popleft()
        if node == forbidden_prefix or node.startswith(f"{forbidden_prefix}."):
            return path
        for child in sorted(graph.get(node, ())):
            if child not in visited:
                visited.add(child)
                pending.append((child, (*path, child)))
    return None


def _cycles(graph: dict[str, set[str]], prefix: str) -> list[tuple[str, ...]]:
    active: list[str] = []
    seen: set[str] = set()
    result: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        seen.add(node)
        active.append(node)
        for child in sorted(graph.get(node, ())):
            if not (child == prefix or child.startswith(f"{prefix}.")):
                continue
            if child in active:
                result.append(tuple((*active[active.index(child) :], child)))
            elif child not in seen:
                visit(child)
        active.pop()

    for node in sorted(graph):
        if (node == prefix or node.startswith(f"{prefix}.")) and node not in seen:
            visit(node)
    return result


def _assert_no_path(
    graph: dict[str, set[str]], start: str, forbidden: str
) -> None:
    path = _find_path(graph, start, forbidden)
    assert path is None, "forbidden transitive import path: " + " -> ".join(path or ())


def test_tome_packaging_and_validation_cannot_reach_builder() -> None:
    graph = _module_graph()
    for start in (
        "radjax_tome.tome.packaging",
        "radjax_tome.tome.producer_validation",
        "radjax_tome.tome.artifact_descriptor",
    ):
        _assert_no_path(graph, start, "radjax_tome.builder")


def test_production_and_domain_layers_cannot_reach_outward_entry_points() -> None:
    graph = _module_graph()
    starts = (
        "radjax_tome.builder.production",
        "radjax_tome.builder.production_stages",
        "radjax_tome.builder.delivery",
        "radjax_tome.artifact_validation",
    )
    forbidden = (
        "radjax_tome.cli",
        "radjax_tome.research",
        "radjax_student",
    )
    for start in starts:
        for blocked in forbidden:
            _assert_no_path(graph, start, blocked)


def test_production_layers_are_acyclic() -> None:
    graph = _module_graph()
    for prefix in (
        "radjax_tome.builder.production_stages",
        "radjax_tome.builder.delivery",
        "radjax_tome.artifact_validation",
    ):
        assert _cycles(graph, prefix) == []


def test_forwarding_module_cannot_hide_a_forbidden_dependency(tmp_path: Path) -> None:
    root = tmp_path / "radjax_tome"
    (root / "tome").mkdir(parents=True)
    (root / "builder").mkdir()
    for path, text in {
        root / "__init__.py": "",
        root / "tome" / "__init__.py": "",
        root / "builder" / "__init__.py": "",
        root / "tome" / "packaging.py": "from radjax_tome.tome import forward\n",
        root / "tome" / "forward.py": "from radjax_tome.builder import leaf\n",
        root / "builder" / "leaf.py": "",
    }.items():
        path.write_text(text, encoding="utf-8")
    graph = _module_graph(root)
    assert _find_path(
        graph, "radjax_tome.tome.packaging", "radjax_tome.builder"
    ) == (
        "radjax_tome.tome.packaging",
        "radjax_tome.tome.forward",
        "radjax_tome.builder",
    )
