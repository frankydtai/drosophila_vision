# -*- coding: utf-8 -*-
"""Moving-bar GT numbers: fig1 Vm traces and motion preference.

GT literals and helpers in this module are **owned here** — fig1 digitized
population Vm (:data:`FIG1_CI_NPZ`) and T4/T5 motion preference.

Network mapping, cost hexes, sti ``i_sti``, and :class:`task.mbar.pack.MbarGt`
packing live in :mod:`task.mbar.pack`. Bar geometry and hex currents live in
:mod:`task.sbar.sti_geo` and :mod:`task.mbar.sti_spec`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from neuron.borst import t_from_ms
from config import MBAR_INPUT_SPEC
from task.mbar.sti_spec import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_MS,
    MbarSpec,
)

# Gruntman Fig. 1 Ci/Cii digitized population Vm (figure_digitization/gruntman21/1ci.py).
# gt.py → mbar → task → simulation → vision → repo root.
FIG1_CI_NPZ = (
    Path(__file__).resolve().parents[4]
    / "figure_digitization"
    / "gruntman21"
    / "1ci_digitized.npz"
)

GT_CELLS: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

GT_CELL_ALIASES: dict = {
    "T4": tuple(cell for cell in GT_CELLS if cell.startswith("T4")),
    "T5": tuple(cell for cell in GT_CELLS if cell.startswith("T5")),
}

_HORIZONTAL = frozenset({"right", "left"})
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}
_SUBTYPE_PD_RIGHT = {"a": "right", "b": "left", "c": "up", "d": "down"}

AXIS_DIRECTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("right", "left"),
    ("up", "down"),
)
_CARDINAL = frozenset(_OPPOSITE)

@dataclass(frozen=True)
class MotionPreference:
    """PD/ND from motion; PC/NC from contrast + pathway (T4 vs T5)."""

    pd_nd: str
    pc_nc: str


def expand_gt_cells(cells: Sequence[str]) -> Tuple[str, ...]:
    """Expand ``--gt`` moving-bar cell tokens via ``GT_CELL_ALIASES`` (e.g. T4, T5)."""
    if not cells:
        raise ValueError("gt_cells must not be empty")
    gt_cells: list = []
    seen: set = set()
    for token in cells:
        token = str(token).strip()
        if token in GT_CELL_ALIASES:
            pool = GT_CELL_ALIASES[token]
        elif token in GT_CELLS:
            pool = (token,)
        else:
            raise ValueError(
                f"unknown gt cell {token!r} (expected {', '.join((*GT_CELL_ALIASES, *GT_CELLS))})"
            )
        for cell in pool:
            if cell not in seen:
                seen.add(cell)
                gt_cells.append(cell)
    return tuple(gt_cells)


def bar_w_token(bar_w_deg: float) -> str:
    return "w1" if float(bar_w_deg) <= 3.0 else "w4"


def pd_direction(side: str, subtype: str) -> str:
    """Preferred-direction motion for ``subtype`` on ``side`` (right or left eye)."""
    letter = subtype[-1]
    if letter not in _SUBTYPE_PD_RIGHT:
        raise ValueError(f"unknown subtype {subtype!r}")
    direction = _SUBTYPE_PD_RIGHT[letter]
    if side == "left" and direction in _HORIZONTAL:
        return _OPPOSITE[direction]
    return direction


def _axis_directions(subtype: str) -> Tuple[str, ...]:
    return AXIS_DIRECTION_PAIRS[0] if subtype[-1] in "ab" else AXIS_DIRECTION_PAIRS[1]


def motion_preference(
    side: str,
    subtype: str,
    direction: str,
    contrast: str,
) -> Optional[MotionPreference]:
    """Map one cardinal sti to PD/ND + PC/NC for a T4/T5 subtype."""
    if subtype not in GT_CELLS:
        raise ValueError(f"unknown subtype {subtype!r}")
    direction = str(direction).strip().lower()
    contrast = str(contrast).strip().lower()
    if direction not in _CARDINAL:
        raise ValueError(f"unknown direction {direction!r}")
    if contrast not in ("bright", "dark"):
        raise ValueError(f"unknown contrast {contrast!r}")
    if direction not in _axis_directions(subtype):
        return None
    pd_dir = pd_direction(side, subtype)
    if direction == pd_dir:
        pd_nd = "PD"
    elif direction == _OPPOSITE[pd_dir]:
        pd_nd = "ND"
    else:
        return None
    pathway = subtype[:2]
    if pathway == "T4":
        pc_nc = "PC" if contrast == "bright" else "NC"
    elif pathway == "T5":
        pc_nc = "PC" if contrast == "dark" else "NC"
    else:
        raise ValueError(f"unknown pathway in {subtype!r}")
    return MotionPreference(pd_nd=pd_nd, pc_nc=pc_nc)


def fig1_trace_from_sti(
    side: str,
    subtype: str,
    spec: Union[MbarSpec, str],
    contrast: Optional[str] = None,
    bar_w_deg: Optional[float] = None,
) -> Optional[str]:
    """fig1 trace token for ``(side, subtype, sti)``, or ``None`` if orthogonal."""
    if isinstance(spec, MbarSpec):
        direction, contrast, bar_w_deg = spec.direction, spec.contrast, spec.bar_w_deg
    else:
        direction = str(spec)
        if contrast is None or bar_w_deg is None:
            raise ValueError("contrast and bar_w_deg required when spec is not MbarSpec")
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return None
    return f"{subtype[:2]}_{pref.pc_nc}_{bar_w_token(bar_w_deg)}_{pref.pd_nd}"


def active_stis_from_subtype(side: str, subtype: str) -> Sequence[Tuple[str, str, str]]:
    """Non-orthogonal (direction, contrast, w_token) triples for one subtype."""
    return [
        (direction, contrast, bar_w_token(bar_w_deg))
        for direction in _axis_directions(subtype)
        for contrast in ("bright", "dark")
        for bar_w_deg in MBAR_INPUT_SPEC["bar_ws_deg"]
        if motion_preference(side, subtype, direction, contrast) is not None
    ]


def parse_mbar_spec(token: str) -> Tuple[str, str, str]:
    direction, contrast, w_token = str(token).split("_", 2)
    return direction, contrast, w_token


_TRACE_CACHE: Dict[str, np.ndarray] = {}


def load_fig1_trace(
    trace_token: str,
    npz_path: Path = FIG1_CI_NPZ,
    *,
    delta_ms: float,
) -> np.ndarray:
    """Resample one fig1 trace onto the moving-bar cost window."""
    n_t = t_from_ms(COST_WINDOW_MS, delta_ms=delta_ms) + 1
    key = f"{trace_token}|{n_t}|{delta_ms}|{COST_WINDOW_MS}|{COST_ALIGNED_FIRST_STI_MS}"
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    with np.load(npz_path) as npz:
        t_key = f"{trace_token}__time_ms"
        if t_key not in npz.files:
            raise KeyError(f"missing trace {trace_token!r} in {npz_path}")
        time_ms = np.asarray(npz[t_key], dtype=np.float64)
        vm_mv = np.asarray(npz[f"{trace_token}__vm_mv"], dtype=np.float64)
    return _TRACE_CACHE.setdefault(
        key,
        np.interp(
            np.arange(n_t, dtype=np.float64) * delta_ms,
            time_ms,
            vm_mv,
            left=vm_mv[0],
            right=vm_mv[-1],
        ),
    )


def load_fig1_traces(
    npz_path: Path = FIG1_CI_NPZ,
    *,
    delta_ms: float,
) -> Dict[str, np.ndarray]:
    """All fig1 traces resampled to the per-hex train window."""
    with np.load(npz_path) as npz:
        trace_tokens = sorted(
            {k.replace("__time_ms", "") for k in npz.files if k.endswith("__time_ms")}
        )
    return {
        trace_token: load_fig1_trace(trace_token, npz_path, delta_ms=delta_ms)
        for trace_token in trace_tokens
    }
