"""Map logical import names to ``N_name`` disk paths under vision.

Roots (tried in order):

  - ``simulation/`` — packages (``neuron.param``, ``network.construction``, …)
  - ``connectome/FAFBv783/`` — flat modules (``build_hex``, ``build_network``, ``add_extent``, …)

Disk names are ``{n}_{logical}`` (dirs and ``.py`` modules). Imports stay
logical. Renumbering is rename-only; this finder has no per-file registry.

Also hosts :func:`parse_comma_list` (sole comma-token splitter for CLI lists),
:func:`parse_bool` (CLI true/false tokens), and :func:`normalize_option_dashes`
(single-dash long options → double-dash; applied to all ``argparse`` parses via
:func:`install`).

``__init__.py`` is unnumbered. Call :func:`install` (or ``import import_bootstrap``)
before any logical import. Project ``.venv`` ``simulation_sorted.pth`` loads this
at interpreter startup.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_SORT_PREFIX = re.compile(r"^\d+_")
_VISION = Path(__file__).resolve().parent
_ROOTS: Tuple[Path, ...] = (
    _VISION / "simulation",
    _VISION / "connectome" / "FAFBv783",
)
_SKIP_NAMES = frozenset({"__pycache__", "0_runs", "0_logs"})
_ORIG_PARSE_KNOWN_ARGS = argparse.ArgumentParser.parse_known_args
_ARGPARSE_DASH_PATCHED = False


def parse_comma_list(text: str) -> List[str]:
    """Split a comma-separated token list (empty string → ``[]``)."""
    return [t.strip() for t in str(text or "").split(",") if t.strip()]


def parse_bool(text) -> bool:
    """Parse CLI boolean (true/false, 1/0, yes/no)."""
    v = str(text).lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"expected true|false, got {text!r}")


def normalize_option_dashes(argv: Sequence[str]) -> List[str]:
    """Rewrite single-dash options to double-dash (``-sti-timing`` → ``--sti-timing``).

    Also rewrites one-letter forms (``-x`` → ``--x``). Leaves ``-h`` (argparse
    help), ``--foo``, ``--``, ``-``, and non-letter bodies (``-1``, ``-0.5``).
    """
    out: List[str] = []
    for tok in argv:
        if (
            isinstance(tok, str)
            and tok.startswith("-")
            and not tok.startswith("--")
            and len(tok) >= 2
        ):
            name = tok[1:].split("=", 1)[0]
            if name and name[0].isalpha() and name != "h":
                tok = "-" + tok
        out.append(tok)
    return out


def _parse_known_args_normalize(self, args=None, namespace=None):
    if args is None:
        args = sys.argv[1:]
    else:
        args = list(args)
    return _ORIG_PARSE_KNOWN_ARGS(self, normalize_option_dashes(args), namespace)


def _install_argparse_dash_normalize() -> None:
    """Make every ArgumentParser accept ``-long-opt`` as ``--long-opt``."""
    global _ARGPARSE_DASH_PATCHED
    if _ARGPARSE_DASH_PATCHED:
        return
    argparse.ArgumentParser.parse_known_args = _parse_known_args_normalize
    _ARGPARSE_DASH_PATCHED = True


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
                f"ambiguous logical name under {directory}: "
                f"multiple children map to {want!r}"
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


class LogicalImportFinder(importlib.abc.MetaPathFinder):
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


_FINDER: Optional[LogicalImportFinder] = None


def install() -> None:
    """Insert the logical-import finder, dash-normalize argparse, and sys.path."""
    global _FINDER
    _install_argparse_dash_normalize()
    if _FINDER is not None and _FINDER in sys.meta_path:
        return
    if _FINDER is None:
        _FINDER = LogicalImportFinder()
    sys.meta_path = [
        x for x in sys.meta_path if not isinstance(x, LogicalImportFinder)
    ]
    sys.meta_path.insert(0, _FINDER)
    for root in (_VISION, *_ROOTS):
        root_s = str(root)
        if root_s not in sys.path:
            sys.path.insert(0, root_s)


install()
