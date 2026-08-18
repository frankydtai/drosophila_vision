# -*- coding: utf-8 -*-
"""Moving-bar GT numbers: fig1 Vm traces, motion preference, DSI targets.

GT literals and helpers in this module are **owned here** — fig1 digitized
population Vm (:data:`FIG1_CI_NPZ`), T4/T5 motion preference, and hardcoded
axis DSI targets (:data:`FIG1_ABS_DSI`).

Network mapping, cost hexes, sti ``i_sti``, and :class:`task.moving_bar.pack.MovingBarGt`
packing live in :mod:`task.moving_bar.pack`. Bar geometry and hex currents live in
:mod:`task.moving_bar.sti_geo` and :mod:`task.moving_bar.sti_spec`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from neuron.borst import t_from_ms
from task.moving_bar.sti_geo import GRUNTMAN_WS_DEG
from task.moving_bar.sti_spec import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_MS,
    MovingBarSpec,
)

# Gruntman Fig. 1 Ci/Cii digitized population Vm (figure_digitization/gruntman21/1ci.py).
# gt.py → moving_bar → task → simulation → vision → repo root.
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
_POS_DIRS = frozenset(pos for pos, _neg in AXIS_DIRECTION_PAIRS)
_DIR_TO_AXIS = {
    **{pos: (pos, neg) for pos, neg in AXIS_DIRECTION_PAIRS},
    **{neg: (pos, neg) for pos, neg in AXIS_DIRECTION_PAIRS},
}

FIG1_ABS_DSI: Mapping[str, float] = {
    "T4_PC_w1": 0.412,
    "T4_PC_w4": 0.514,
    "T4_NC_w1": 0.024,
    "T4_NC_w4": 0.389,
    "T5_PC_w1": 0.322,
    "T5_PC_w4": 0.469,
    "T5_NC_w1": 0.263,
    "T5_NC_w4": 0.246,
}


@dataclass(frozen=True)
class MotionPreference:
    """PD/ND from motion; PC/NC from contrast + pathway (T4 vs T5)."""

    pd_nd: str
    pc_nc: str


def dsi_sequential_b_sets(spec_tokens: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    """Minimal sti-b sets for sequential DSI: one b_set per axis x w."""
    bs_by_dir_w: dict[tuple[str, str], list[int]] = {}
    for b, token in enumerate(spec_tokens):
        direction, _contrast, w_token = parse_moving_bar_spec(token)
        bs_by_dir_w.setdefault((direction, w_token), []).append(int(b))
    b_sets: list[tuple[int, ...]] = []
    for pos_dir, neg_dir in AXIS_DIRECTION_PAIRS:
        w_tokens = {
            w_token for (direction, w_token) in bs_by_dir_w
            if direction in (pos_dir, neg_dir)
        }
        for w_token in sorted(w_tokens):
            pos = bs_by_dir_w.get((pos_dir, w_token), [])
            neg = bs_by_dir_w.get((neg_dir, w_token), [])
            if not pos or not neg:
                continue
            b_sets.append(tuple(sorted({*pos, *neg})))
    return tuple(b_sets)


def expand_gt_cells(cells: Sequence[str]) -> Tuple[str, ...]:
    """Expand ``--gt`` moving-bar cell tokens via ``GT_CELL_ALIASES`` (e.g. T4, T5)."""
    if not cells:
        raise ValueError("gt_cells must not be empty")
    out: list = []
    seen: set = set()
    for token in cells:
        token = str(token).strip()
        if token in GT_CELL_ALIASES:
            pool = GT_CELL_ALIASES[token]
        elif token in GT_CELLS:
            pool = (token,)
        else:
            valid = ", ".join((*GT_CELL_ALIASES, *GT_CELLS))
            raise ValueError(f"unknown gt cell {token!r} (expected {valid})")
        for cell in pool:
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
    return tuple(out)


def w_token(w_deg: float) -> str:
    return "w1" if float(w_deg) <= 3.0 else "w4"


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
    spec: Union[MovingBarSpec, str],
    contrast: Optional[str] = None,
    w_deg: Optional[float] = None,
) -> Optional[str]:
    """fig1 trace token for ``(side, subtype, sti)``, or ``None`` if orthogonal."""
    if isinstance(spec, MovingBarSpec):
        direction, contrast, w_deg = spec.direction, spec.contrast, spec.w_deg
    else:
        direction = str(spec)
        if contrast is None or w_deg is None:
            raise ValueError("contrast and w_deg required when spec is not MovingBarSpec")
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return None
    return f"{subtype[:2]}_{pref.pc_nc}_{w_token(w_deg)}_{pref.pd_nd}"


def active_stis_from_subtype(side: str, subtype: str) -> Sequence[Tuple[str, str, str]]:
    """Non-orthogonal (direction, contrast, w_token) triples for one subtype."""
    out = []
    for direction in _axis_directions(subtype):
        for contrast in ("bright", "dark"):
            for w_deg in GRUNTMAN_WS_DEG:
                if motion_preference(side, subtype, direction, contrast) is not None:
                    out.append((direction, contrast, w_token(w_deg)))
    return out


def parse_moving_bar_spec(token: str) -> Tuple[str, str, str]:
    direction, contrast, w_token = str(token).split("_", 2)
    return direction, contrast, w_token


def axis_dsi(peak_pos: float, peak_neg: float) -> Optional[float]:
    """DSI = (peak_pos - peak_neg) / (peak_pos + peak_neg); pos = right|up."""
    denom = float(peak_pos) + float(peak_neg)
    if denom <= 0.0:
        return None
    return (float(peak_pos) - float(peak_neg)) / denom


def axis_dsi_torch(peak_pos: torch.Tensor, peak_neg: torch.Tensor) -> torch.Tensor:
    """Elementwise axis DSI; denom <= 0 -> 0."""
    denom = peak_pos + peak_neg
    return torch.where(denom > 0, (peak_pos - peak_neg) / denom, torch.zeros_like(denom))


def hardcoded_axis_dsi(side: str, subtype: str, spec: MovingBarSpec) -> Optional[float]:
    """Signed axis DSI from ``FIG1_ABS_DSI`` for the pos-side sti ``spec``."""
    pos_trace = fig1_trace_from_sti(side, subtype, spec)
    if pos_trace is None:
        return None
    base, pd_nd = pos_trace.rsplit("_", 1)
    if base not in FIG1_ABS_DSI:
        raise KeyError(f"hardcoded DSI missing: {base}")
    abs_dsi = float(FIG1_ABS_DSI[base])
    if pd_nd == "PD":
        return abs_dsi
    if pd_nd == "ND":
        return -abs_dsi
    raise ValueError(f"expected PD/ND suffix in {pos_trace!r}")


def moving_bar_dsi_from_spec(
    trace_map: Mapping[tuple, np.ndarray],
    cell: str,
    token: str,
) -> Optional[float]:
    """DSI for one cell x spec: (this dir - opposite) / (this + opposite)."""
    direction, contrast, w_token = parse_moving_bar_spec(token)
    if direction not in _DIR_TO_AXIS:
        return None
    pos_dir, neg_dir = _DIR_TO_AXIS[direction]
    pos_trace = (cell, f"{pos_dir}_{contrast}_{w_token}")
    neg_key = (cell, f"{neg_dir}_{contrast}_{w_token}")
    if pos_trace not in trace_map or neg_key not in trace_map:
        return None
    dsi = axis_dsi(
        float(np.max(np.asarray(trace_map[pos_trace], dtype=np.float64))),
        float(np.max(np.asarray(trace_map[neg_key], dtype=np.float64))),
    )
    if dsi is None:
        return None
    if direction in _POS_DIRS:
        return dsi
    return -dsi


def dsi_from_trace_map(
    trace_map: Mapping[tuple, np.ndarray],
    cells: Sequence[str],
    spec_tokens: Sequence[str],
) -> dict[tuple[str, str], Optional[float]]:
    out: dict[tuple[str, str], Optional[float]] = {}
    for cell in cells:
        for spec in spec_tokens:
            key = (cell, spec)
            if key not in out:
                out[key] = moving_bar_dsi_from_spec(trace_map, cell, spec)
    return out


def moving_bar_cell_title(
    head: str,
    ca_dsi: Optional[float] = None,
    gt_dsi: Optional[float] = None,
    *,
    has_gt: bool = False,
) -> str:
    """Append DSI lines to a subplot title *head*."""
    lines = [str(head)]
    if ca_dsi is not None:
        lines.append(f"DSI={ca_dsi:.3f}")
    if has_gt and gt_dsi is not None:
        lines.append(f"gt DSI={gt_dsi:.3f}")
    return "\n".join(lines)


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
        t_key, v_key = f"{trace_token}__time_ms", f"{trace_token}__vm_mv"
        if t_key not in npz.files:
            raise KeyError(f"missing trace {trace_token!r} in {npz_path}")
        time_ms = np.asarray(npz[t_key], dtype=np.float64)
        vm_mv = np.asarray(npz[v_key], dtype=np.float64)
    query_ms = np.arange(n_t, dtype=np.float64) * delta_ms
    trace = np.interp(query_ms, time_ms, vm_mv, left=vm_mv[0], right=vm_mv[-1])
    _TRACE_CACHE[key] = trace
    return trace


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
