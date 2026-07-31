"""Map logical import names to ``N_name`` disk layout under vision.

Roots (tried in order):

  - ``simulation/`` — packages (``neuron.params``, ``network.layout``, …)
  - ``connectome/FAFBv783/`` — flat modules (``build_hex``, ``build_network``, …)

Disk names are ``{n}_{logical}`` (dirs and ``.py`` modules). Imports stay
logical. Renumbering is rename-only; this finder has no per-file registry.

``__init__.py`` is unnumbered. Call :func:`install` (or ``import import_bootstrap``)
before any logical import. Project ``.venv`` ``simulation_sorted.pth`` loads this
at interpreter startup.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

_SORT_PREFIX = re.compile(r"^\d+_")
_VISION = Path(__file__).resolve().parent
_ROOTS: Tuple[Path, ...] = (
    _VISION / "simulation",
    _VISION / "connectome" / "FAFBv783",
)
_SKIP_NAMES = frozenset({"__pycache__", "0_runs", "0_logs"})


def logical_name(name: str) -> str:
    """Strip leading ``digits_`` from a directory name or file stem."""
    return _SORT_PREFIX.sub("", name)


def _iter_children(directory: Path) -> Iterable[Path]:
    try:
        children = list(directory.iterdir())
    except OSError:
        return
    for child in children:
        if child.name.startswith(".") or child.name in _SKIP_NAMES:
            continue
        yield child


def child_by_logical(directory: Path, want: str) -> Optional[Path]:
    """Return the child whose logical name equals ``want``, or None."""
    matched: Optional[Path] = None
    for child in _iter_children(directory):
        if child.name == "__init__.py":
            continue
        key = logical_name(child.stem if child.is_file() else child.name)
        if key != want:
            continue
        if matched is not None:
            raise ImportError(
                f"ambiguous sorted layout under {directory}: "
                f"multiple children map to logical name {want!r}"
            )
        matched = child
    return matched


def resolve_parts_under(root: Path, parts: Sequence[str]) -> Optional[Path]:
    """Resolve logical dotted parts under one root."""
    if not parts:
        return None
    cur = root
    for i, part in enumerate(parts):
        hit = child_by_logical(cur, part)
        if hit is None:
            return None
        last = i == len(parts) - 1
        if last:
            return hit
        if not hit.is_dir():
            return None
        cur = hit
    return None


def resolve_parts(parts: Sequence[str]) -> Optional[Path]:
    """Resolve logical dotted parts under the first matching root."""
    for root in _ROOTS:
        hit = resolve_parts_under(root, parts)
        if hit is not None:
            return hit
    return None


class SortedLayoutFinder(importlib.abc.MetaPathFinder):
    """Load ``logical`` names from ``N_logical`` paths under vision roots."""

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001
        del path, target
        parts = fullname.split(".")
        hit = resolve_parts(parts)
        if hit is None:
            return None

        if hit.is_file():
            if hit.suffix != ".py":
                return None
            return importlib.util.spec_from_file_location(fullname, hit)

        if not hit.is_dir():
            return None

        init = hit / "__init__.py"
        if init.is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                init,
                submodule_search_locations=[str(hit)],
            )
        return importlib.machinery.ModuleSpec(
            fullname,
            loader=None,
            is_package=True,
        )


_FINDER: Optional[SortedLayoutFinder] = None


def install() -> None:
    """Insert the sorted-layout finder and ensure roots are on ``sys.path``."""
    global _FINDER
    if _FINDER is not None and _FINDER in sys.meta_path:
        return
    if _FINDER is None:
        _FINDER = SortedLayoutFinder()
    sys.meta_path = [
        x for x in sys.meta_path if not isinstance(x, SortedLayoutFinder)
    ]
    sys.meta_path.insert(0, _FINDER)
    for root in (_VISION, *_ROOTS):
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)


install()
