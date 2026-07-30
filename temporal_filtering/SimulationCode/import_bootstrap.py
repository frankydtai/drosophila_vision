"""Map logical import names to ``N_name`` disk layout under SimulationCode.

Disk names are ``{n}_{logical}`` (dirs and ``.py`` modules). Imports stay
logical (``neuron.params``, ``task.spot.input``, …). Renumbering is
rename-only; this finder has no per-file registry.

``__init__.py`` is unnumbered. Call :func:`install` (or ``import import_bootstrap``)
before any logical package import.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

_SORT_PREFIX = re.compile(r"^\d+_")
_ROOT = Path(__file__).resolve().parent
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


def resolve_parts(parts: Sequence[str]) -> Optional[Path]:
    """Resolve logical dotted parts to a file or package directory under ROOT."""
    if not parts:
        return None
    cur = _ROOT
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


class SortedLayoutFinder(importlib.abc.MetaPathFinder):
    """Load ``logical`` packages from ``N_logical`` paths under SimulationCode."""

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
        # Namespace package (no __init__.py)
        return importlib.machinery.ModuleSpec(
            fullname,
            loader=None,
            is_package=True,
        )


_FINDER: Optional[SortedLayoutFinder] = None


def install() -> None:
    """Insert the sorted-layout finder at the front of ``sys.meta_path`` once."""
    global _FINDER
    if _FINDER is not None and _FINDER in sys.meta_path:
        return
    if _FINDER is None:
        _FINDER = SortedLayoutFinder()
    # Drop a stale instance after reload
    sys.meta_path = [
        x for x in sys.meta_path if not isinstance(x, SortedLayoutFinder)
    ]
    sys.meta_path.insert(0, _FINDER)
    root_s = str(_ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


install()
