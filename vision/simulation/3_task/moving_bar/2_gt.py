# -*- coding: utf-8 -*-
"""Moving-bar paradigm GT: fig1 Vm gt, T4/T5 DSI cost, and stimulus opts.

Merges the old ``t4_t5_dsi`` (motion preference + DSI math) and the gt /
cost-readout half of ``network.moving_bar_readout`` (fig1 traces, cost windows,
DSI pack fields), plus the moving-bar ``stimulus_opts`` builders from the old
``FiveCol_MedSim_Pytorch``. Signals / geometry / t0 grids live in
:mod:`task.moving_bar.input`; this module imports downward from it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from neuron.params import ms_to_t
from network.construction import (
    col2gt,
    present_gt_cells,
    normalize_gt_cells,
    node_cell_names,
)
from task.moving_bar.input import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_AFTER_MS,
    COST_WINDOW_BEFORE_MS,
    COST_WINDOW_MS,
    DEFAULT_BAR_EXTENT,
    ND_IDX,
    PD_IDX,
    MovingBarSpec,
    build_moving_bar_signals,
    build_moving_bar_t0_grids,
    hex_first_stim_t,
    filter_sti_hexes,
    gruntman_moving_bar_specs,
    moving_bar_cost_hexes,
    moving_bar_i_baseline_from_opts,
    network_uv_np,
    resolve_i_baseline,
    sti_hexes,
    _as_int64_np,
)

# Gruntman Fig. 1 Ci/Cii digitized population Vm (MatlabFunctions/digitize_fig1_ci.py).
# gt.py → moving_bar → task → simulation → vision → repo root.
FIG1_CI_NPZ = (
    Path(__file__).resolve().parents[4] / "MatlabFunctions" / "fig1_ci_digitized.npz"
)
from task.moving_bar.input import GRUNTMAN_WIDTHS_DEG

logger = logging.getLogger(__name__)

MOVING_BAR_POLARITIES = frozenset({"bright", "dark"})


# =====================================================================
# T4 / T5 motion preference + DSI (was t4_t5_dsi)
# =====================================================================

GT_CELLS: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

GT_CELL_ALIASES: dict = {
    "T4": tuple(st for st in GT_CELLS if st.startswith("T4")),
    "T5": tuple(st for st in GT_CELLS if st.startswith("T5")),
}

_HORIZONTAL = frozenset({"right", "left"})
_VERTICAL = frozenset({"up", "down"})
_HORIZONTAL_AXIS: Tuple[str, ...] = ("right", "left")
_VERTICAL_AXIS: Tuple[str, ...] = ("up", "down")
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}
_SUBTYPE_PD_RIGHT = {"a": "right", "b": "left", "c": "up", "d": "down"}

AXIS_DIRECTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("right", "left"),
    ("up", "down"),
)
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


def dsi_sequential_batch_groups(spec_names: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    """Minimal stimulus-batch groups for sequential DSI: one group per axis x width."""
    batches_by_dir_w: dict[tuple[str, str], list[int]] = {}
    for bi, sname in enumerate(spec_names):
        direction, _contrast, wtag = parse_moving_bar_spec(sname)
        batches_by_dir_w.setdefault((direction, wtag), []).append(int(bi))
    groups: list[tuple[int, ...]] = []
    for pos_dir, neg_dir in AXIS_DIRECTION_PAIRS:
        wtags = {
            wtag for (direction, wtag) in batches_by_dir_w
            if direction in (pos_dir, neg_dir)
        }
        for wtag in sorted(wtags):
            pos = batches_by_dir_w.get((pos_dir, wtag), [])
            neg = batches_by_dir_w.get((neg_dir, wtag), [])
            if not pos or not neg:
                continue
            groups.append(tuple(sorted({*pos, *neg})))
    return tuple(groups)


def expand_gt_cells(names: Sequence[str]) -> Tuple[str, ...]:
    """Expand ``--gt`` moving-bar cell tokens via ``GT_CELL_ALIASES`` (e.g. T4, T5)."""
    if not names:
        raise ValueError("gt_cells must not be empty")
    out: list = []
    seen: set = set()
    for raw in names:
        key = str(raw).strip()
        if key in GT_CELL_ALIASES:
            pool = GT_CELL_ALIASES[key]
        elif key in GT_CELLS:
            pool = (key,)
        else:
            valid = ", ".join((*GT_CELL_ALIASES, *GT_CELLS))
            raise ValueError(f"unknown gt cell {key!r} (expected {valid})")
        for st in pool:
            if st not in seen:
                seen.add(st)
                out.append(st)
    return tuple(out)


def normalize_side(side: str) -> str:
    s = str(side).strip().lower()
    if s in ("r", "right"):
        return "right"
    if s in ("l", "left"):
        return "left"
    raise ValueError(f"unknown eye side {side!r}")


def width_tag(width_deg: float) -> str:
    return "w1" if float(width_deg) <= 3.0 else "w4"


def width_tag_from_deg(width_deg: float) -> str:
    return width_tag(width_deg)


def pd_direction(side: str, subtype: str) -> str:
    """Preferred-direction motion for ``subtype`` on ``side`` (right or left eye)."""
    side = normalize_side(side)
    letter = subtype[-1]
    if letter not in _SUBTYPE_PD_RIGHT:
        raise ValueError(f"unknown subtype {subtype!r}")
    d = _SUBTYPE_PD_RIGHT[letter]
    if side == "left" and d in _HORIZONTAL:
        return _OPPOSITE[d]
    return d


def _axis_directions(subtype: str) -> Tuple[str, ...]:
    return _HORIZONTAL_AXIS if subtype[-1] in "ab" else _VERTICAL_AXIS


def motion_preference(
    side: str,
    subtype: str,
    direction: str,
    contrast: str,
) -> Optional[MotionPreference]:
    """Map one cardinal stimulus to PD/ND + PC/NC for a T4/T5 subtype."""
    if subtype not in GT_CELLS:
        raise ValueError(f"unknown subtype {subtype!r}")
    direction = str(direction).strip().lower()
    contrast = str(contrast).strip().lower()
    if direction not in _HORIZONTAL | _VERTICAL:
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


def fig1_trace_key(pathway: str, pc_nc: str, width: str, pd_nd: str) -> str:
    """Key in ``fig1_ci_digitized.npz`` (e.g. ``T4_PC_w1_PD``)."""
    return f"{pathway}_{pc_nc}_{width}_{pd_nd}"


def fig1_key_for_stimulus(
    side: str,
    subtype: str,
    spec: Union[MovingBarSpec, str],
    contrast: Optional[str] = None,
    width_deg: Optional[float] = None,
) -> Optional[str]:
    """fig1 trace id for ``(side, subtype, stimulus)``, or ``None`` if orthogonal."""
    if isinstance(spec, MovingBarSpec):
        direction, contrast, width_deg = spec.direction, spec.contrast, spec.width_deg
    else:
        direction = str(spec)
        if contrast is None or width_deg is None:
            raise ValueError("contrast and width_deg required when spec is not MovingBarSpec")
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return None
    return fig1_trace_key(subtype[:2], pref.pc_nc, width_tag(width_deg), pref.pd_nd)


def active_stimuli_for_subtype(side: str, subtype: str) -> Sequence[Tuple[str, str, str]]:
    """Non-orthogonal (direction, contrast, width_tag) triples for one subtype."""
    out = []
    for direction in _axis_directions(subtype):
        for contrast in ("bright", "dark"):
            for w in GRUNTMAN_WIDTHS_DEG:
                if motion_preference(side, subtype, direction, contrast) is not None:
                    out.append((direction, contrast, width_tag(w)))
    return out


def parse_moving_bar_spec(sname: str) -> Tuple[str, str, str]:
    direction, contrast, wtag = str(sname).split("_", 2)
    return direction, contrast, wtag


def trace_peak(trace: np.ndarray) -> float:
    return float(np.max(np.asarray(trace, dtype=np.float64)))


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


def _hardcoded_axis_dsi(side: str, subtype: str, spec: MovingBarSpec) -> Optional[float]:
    """Signed axis DSI from ``FIG1_ABS_DSI`` for the pos-side stimulus ``spec``."""
    pos_key = fig1_key_for_stimulus(side, subtype, spec)
    if pos_key is None:
        return None
    base, pd_nd = pos_key.rsplit("_", 1)
    if base not in FIG1_ABS_DSI:
        raise KeyError(f"hardcoded DSI missing: {base}")
    abs_dsi = float(FIG1_ABS_DSI[base])
    if pd_nd == "PD":
        return abs_dsi
    if pd_nd == "ND":
        return -abs_dsi
    raise ValueError(f"expected PD/ND suffix in {pos_key!r}")


def assemble_moving_bar_dsi_groups(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_subtype: Sequence[str],
    r_weight: Sequence[float],
    *,
    side: str,
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """One DSI group per ``(subtype, contrast, wtag, axis)``."""
    batches_by_condition: dict[tuple[str, str, str], list[int]] = {}
    for bi, spec in enumerate(specs):
        key = (spec.direction, spec.contrast, width_tag(spec.width_deg))
        batches_by_condition.setdefault(key, []).append(bi)

    rows_by_subtype_batch: dict[tuple[str, int], list[int]] = {}
    for i, (b, st) in enumerate(zip(r_batch, r_subtype)):
        rows_by_subtype_batch.setdefault((str(st), int(b)), []).append(i)

    pos_groups: List[List[int]] = []
    neg_groups: List[List[int]] = []
    dsi_vals: List[float] = []
    weights: List[float] = []
    subtypes = sorted({str(st) for st in r_subtype})
    for subtype in subtypes:
        for pos_dir, neg_dir in AXIS_DIRECTION_PAIRS:
            contrast_widths = {
                (contrast, wtag)
                for (direction, contrast, wtag) in batches_by_condition
                if direction in (pos_dir, neg_dir)
            }
            for contrast, wtag in sorted(contrast_widths):
                pos_batches = batches_by_condition.get((pos_dir, contrast, wtag), [])
                neg_batches = batches_by_condition.get((neg_dir, contrast, wtag), [])
                if not pos_batches or not neg_batches:
                    continue
                pos_rows: list[int] = []
                for pb in pos_batches:
                    pos_rows.extend(rows_by_subtype_batch.get((subtype, pb), []))
                neg_rows: list[int] = []
                for nb in neg_batches:
                    neg_rows.extend(rows_by_subtype_batch.get((subtype, nb), []))
                if not pos_rows or not neg_rows:
                    continue
                dsi = _hardcoded_axis_dsi(side, subtype, specs[pos_batches[0]])
                if dsi is None:
                    continue
                w_pos = float(np.mean([float(r_weight[i]) for i in pos_rows]))
                w_neg = float(np.mean([float(r_weight[i]) for i in neg_rows]))
                pos_groups.append(pos_rows)
                neg_groups.append(neg_rows)
                dsi_vals.append(float(dsi))
                weights.append(0.5 * (w_pos + w_neg))
    return pos_groups, neg_groups, dsi_vals, weights


def _csr_from_groups(
    groups: Sequence[Sequence[int]], *, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(flat_rows, ptr)`` with ``ptr`` length ``n_groups + 1``."""
    ptr = [0]
    flat: list[int] = []
    for g in groups:
        flat.extend(int(i) for i in g)
        ptr.append(len(flat))
    rows_t = torch.tensor(np.asarray(flat, dtype=np.int64), dtype=torch.long, device=device)
    ptr_t = torch.tensor(np.asarray(ptr, dtype=np.int64), dtype=torch.long, device=device)
    return rows_t, ptr_t


def pack_moving_bar_dsi_tensors(pos_groups, neg_groups, dsi_vals, weights, *, device, sim_dtype):
    if not pos_groups:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros(0, dtype=sim_dtype, device=device)
        ptr0 = torch.zeros(1, dtype=torch.long, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return empty_long, empty_long, ptr0, ptr0, empty, empty, power
    dsi_pos_rows, dsi_pos_ptr = _csr_from_groups(pos_groups, device=device)
    dsi_neg_rows, dsi_neg_ptr = _csr_from_groups(neg_groups, device=device)
    dsi_gt = torch.tensor(np.asarray(dsi_vals), dtype=sim_dtype, device=device)
    dsi_weight = torch.tensor(np.asarray(weights), dtype=sim_dtype, device=device)
    power = torch.sum(dsi_weight * dsi_gt ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return (
        dsi_pos_rows, dsi_neg_rows, dsi_pos_ptr, dsi_neg_ptr,
        dsi_gt, dsi_weight, power,
    )


def build_dsi_pack_fields(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_subtype: Sequence[str],
    r_weight: Sequence[float],
    *,
    side: str,
    device,
    sim_dtype,
):
    """Assemble + tensorize subtype-grouped DSI fields for a moving-bar task."""
    return pack_moving_bar_dsi_tensors(
        *assemble_moving_bar_dsi_groups(specs, r_batch, r_subtype, r_weight, side=side),
        device=device,
        sim_dtype=sim_dtype,
    )


def _empty_dsi_fields(pack, device) -> dict:
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    empty = torch.zeros(0, dtype=pack.dsi_gt.dtype, device=device)
    ptr0 = torch.zeros(1, dtype=torch.long, device=device)
    power = torch.tensor(1.0, dtype=pack.dsi_power.dtype, device=device)
    return {
        "dsi_pos_rows": empty_long,
        "dsi_neg_rows": empty_long,
        "dsi_pos_ptr": ptr0,
        "dsi_neg_ptr": ptr0,
        "dsi_gt": empty,
        "dsi_weight": empty,
        "dsi_power": power,
    }


def remap_dsi_rows(pack, kept_old_rows) -> dict:
    """Remap CSR DSI members onto kept cost-row indices; drop incomplete groups."""
    if pack.dsi_pos_ptr is None or int(pack.dsi_pos_ptr.numel()) <= 1:
        return {
            "dsi_pos_rows": pack.dsi_pos_rows,
            "dsi_neg_rows": pack.dsi_neg_rows,
            "dsi_pos_ptr": pack.dsi_pos_ptr,
            "dsi_neg_ptr": pack.dsi_neg_ptr,
            "dsi_gt": pack.dsi_gt,
            "dsi_weight": pack.dsi_weight,
            "dsi_power": pack.dsi_power,
        }
    device = pack.dsi_pos_rows.device
    n = int(pack.readout_batch.shape[0])
    kept = torch.as_tensor(kept_old_rows, dtype=torch.long, device=device)
    lut = torch.full((n,), -1, dtype=torch.long, device=device)
    lut[kept] = torch.arange(kept.numel(), dtype=torch.long, device=device)

    n_dsi = int(pack.dsi_pos_ptr.numel()) - 1
    new_pos_groups: list[list[int]] = []
    new_neg_groups: list[list[int]] = []
    keep_g: list[int] = []
    pos_rows = pack.dsi_pos_rows
    neg_rows = pack.dsi_neg_rows
    pos_ptr = pack.dsi_pos_ptr
    neg_ptr = pack.dsi_neg_ptr
    for g in range(n_dsi):
        p0, p1 = int(pos_ptr[g]), int(pos_ptr[g + 1])
        n0, n1 = int(neg_ptr[g]), int(neg_ptr[g + 1])
        new_pos = lut[pos_rows[p0:p1]]
        new_neg = lut[neg_rows[n0:n1]]
        new_pos = new_pos[new_pos >= 0]
        new_neg = new_neg[new_neg >= 0]
        if new_pos.numel() == 0 or new_neg.numel() == 0:
            continue
        new_pos_groups.append(new_pos.tolist())
        new_neg_groups.append(new_neg.tolist())
        keep_g.append(g)
    if not keep_g:
        return _empty_dsi_fields(pack, device)
    dsi_pos_rows, dsi_pos_ptr = _csr_from_groups(new_pos_groups, device=device)
    dsi_neg_rows, dsi_neg_ptr = _csr_from_groups(new_neg_groups, device=device)
    ix = torch.tensor(keep_g, dtype=torch.long, device=device)
    dsi_gt = pack.dsi_gt[ix]
    dsi_weight = pack.dsi_weight[ix]
    power = torch.sum(dsi_weight * dsi_gt ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=dsi_gt.dtype, device=device)
    return {
        "dsi_pos_rows": dsi_pos_rows,
        "dsi_neg_rows": dsi_neg_rows,
        "dsi_pos_ptr": dsi_pos_ptr,
        "dsi_neg_ptr": dsi_neg_ptr,
        "dsi_gt": dsi_gt,
        "dsi_weight": dsi_weight,
        "dsi_power": power,
    }


def _csr_group_mean(values: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over CSR groups defined by ``ptr``."""
    n_g = int(ptr.numel()) - 1
    if n_g == 0:
        return values.new_zeros((0,))
    counts = ptr[1:] - ptr[:-1]
    gid = torch.repeat_interleave(
        torch.arange(n_g, device=values.device, dtype=torch.long),
        counts,
    )
    sums = values.new_zeros((n_g,))
    sums.scatter_add_(0, gid, values)
    return sums / counts.to(dtype=values.dtype).clamp(min=1)


def cost_dsi_from_sel(pack, scale: torch.Tensor, sel: torch.Tensor) -> Optional[torch.Tensor]:
    """Unweighted DSI MSE (% of dsi_power); None if no complete groups."""
    if pack.dsi_pos_ptr is None or int(pack.dsi_pos_ptr.numel()) <= 1:
        return None
    peak_pos_u = (scale[pack.dsi_pos_rows, None] * sel[pack.dsi_pos_rows]).amax(dim=-1)
    peak_neg_u = (scale[pack.dsi_neg_rows, None] * sel[pack.dsi_neg_rows]).amax(dim=-1)
    if not (torch.isfinite(peak_pos_u).all() and torch.isfinite(peak_neg_u).all()):
        raise RuntimeError("non-finite DSI peaks (NaN/Inf in readout)")
    peak_pos = _csr_group_mean(peak_pos_u, pack.dsi_pos_ptr)
    peak_neg = _csr_group_mean(peak_neg_u, pack.dsi_neg_ptr)
    model_dsi = axis_dsi_torch(peak_pos, peak_neg)
    diff = model_dsi - pack.dsi_gt
    return torch.sum(pack.dsi_weight * diff ** 2) / pack.dsi_power * 100.0


def moving_bar_dsi_for_spec(
    trace_map: Mapping[tuple, np.ndarray],
    cell_name: str,
    spec_name: str,
) -> Optional[float]:
    """DSI for one cell x spec: (this dir - opposite) / (this + opposite)."""
    direction, contrast, wtag = parse_moving_bar_spec(spec_name)
    if direction not in _DIR_TO_AXIS:
        return None
    pos_dir, neg_dir = _DIR_TO_AXIS[direction]
    pos_key = (cell_name, f"{pos_dir}_{contrast}_{wtag}")
    neg_key = (cell_name, f"{neg_dir}_{contrast}_{wtag}")
    if pos_key not in trace_map or neg_key not in trace_map:
        return None
    dsi = axis_dsi(trace_peak(trace_map[pos_key]), trace_peak(trace_map[neg_key]))
    if dsi is None:
        return None
    if direction in _POS_DIRS:
        return dsi
    return -dsi


def moving_bar_dsi_lookup(
    trace_map: Mapping[tuple, np.ndarray],
    cell_names: Sequence[str],
    spec_names: Sequence[str],
) -> dict[tuple[str, str], Optional[float]]:
    out: dict[tuple[str, str], Optional[float]] = {}
    for cell in cell_names:
        for spec in spec_names:
            key = (cell, spec)
            if key not in out:
                out[key] = moving_bar_dsi_for_spec(trace_map, cell, spec)
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


# =====================================================================
# fig1 Vm gt + T4/T5 cost readouts (was network.moving_bar_readout)
# =====================================================================

_TRACE_CACHE: Dict[str, np.ndarray] = {}


def _pd_nd_index(pd_nd: str) -> int:
    return PD_IDX if pd_nd == "PD" else ND_IDX


@dataclass
class MovingBarGt:
    i_sti: torch.Tensor
    gt: torch.Tensor
    power: torch.Tensor
    cost_weight: torch.Tensor
    readout_batch: torch.Tensor
    readout_node: torch.Tensor
    n_batch: int
    n_t: int
    info: dict
    cost_t0: Optional[torch.Tensor] = None
    cost_pd_nd: Optional[torch.Tensor] = None
    dsi_pos_rows: Optional[torch.Tensor] = None
    dsi_neg_rows: Optional[torch.Tensor] = None
    dsi_pos_ptr: Optional[torch.Tensor] = None
    dsi_neg_ptr: Optional[torch.Tensor] = None
    dsi_gt: Optional[torch.Tensor] = None
    dsi_weight: Optional[torch.Tensor] = None
    dsi_power: Optional[torch.Tensor] = None


def _fig1_trace_ids(npz_path: Path) -> List[str]:
    with np.load(npz_path) as d:
        return sorted({k.replace("__time_ms", "") for k in d.files if k.endswith("__time_ms")})


def load_fig1_trace(
    trace_id: str,
    npz_path: Path = FIG1_CI_NPZ,
    *,
    delta_ms: float) -> np.ndarray:
    """Resample one fig1 trace onto the moving-bar cost window."""
    n_t = ms_to_t(COST_WINDOW_MS, delta_ms=delta_ms) + 1
    before_t = ms_to_t(COST_ALIGNED_FIRST_STI_MS, delta_ms=delta_ms)
    key = (
        f"{trace_id}|{n_t}|{before_t}|{delta_ms}"
        f"|{COST_WINDOW_MS}|{COST_ALIGNED_FIRST_STI_MS}"
    )
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]
    with np.load(npz_path) as d:
        t_key, v_key = f"{trace_id}__time_ms", f"{trace_id}__vm_mv"
        if t_key not in d.files:
            raise KeyError(f"missing trace {trace_id!r} in {npz_path}")
        time_ms = np.asarray(d[t_key], dtype=np.float64)
        vm_mv = np.asarray(d[v_key], dtype=np.float64)
    query_ms = np.arange(n_t, dtype=np.float64) * delta_ms
    trace = np.interp(query_ms, time_ms, vm_mv, left=vm_mv[0], right=vm_mv[-1])
    _TRACE_CACHE[key] = trace
    return trace


def load_fig1_traces(
    npz_path: Path = FIG1_CI_NPZ,
    *,
    delta_ms: float) -> Dict[str, np.ndarray]:
    """All fig1 traces resampled to the per-hex training window."""
    return {
        tid: load_fig1_trace(tid, npz_path, delta_ms)
        for tid in _fig1_trace_ids(npz_path)
    }


def moving_bar_nodes_on_hexes(C, cell: str, hexes: Sequence) -> np.ndarray:
    """Node indices of ``cell`` on any of ``hexes`` (vectorized axial uv pack)."""
    if not hexes:
        return np.zeros(0, dtype=np.int64)
    if cell not in C.cell_names:
        raise ValueError(f"unknown cell {cell!r}; known: {list(C.cell_names)}")
    ti = int(C.cell_names.index(cell))
    u_np, v_np = network_uv_np(C)
    cell_ids = _as_int64_np(C.node_cell)
    uv_span = int(max(np.max(np.abs(u_np)), np.max(np.abs(v_np)), 1)) + 1
    pack = (u_np + uv_span) * (2 * uv_span + 1) + (v_np + uv_span)
    hex_pack = np.array(
        [
            (int(c.u) + uv_span) * (2 * uv_span + 1) + (int(c.v) + uv_span)
            for c in hexes
        ],
        dtype=np.int64,
    )
    return np.where((cell_ids == ti) & np.isin(pack, hex_pack))[0].astype(np.int64)


def filter_requested_specs(
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> List[str]:
    """Keep ``requested`` specs that exist in ``available``; omit ``requested`` -> all."""
    avail = list(available)
    if requested is None:
        return avail
    missing = [s for s in requested if s not in avail]
    if missing:
        raise ValueError(f"spec(s) {missing} not in {avail}")
    return list(requested)


def _assemble_moving_bar_readouts(
    *,
    specs: Sequence[MovingBarSpec],
    hex_current: np.ndarray,
    cost_hex_idxs: Sequence[int],
    i_baseline: float,
    before_t: int,
    after_t: int,
    n_t: int,
    side: str,
    fig1: Optional[Dict[str, np.ndarray]],
    present: Sequence[str],
    nodes_for_hex_type: Callable[[int, int, str], Sequence[int]],
    waveform_mse: bool = True,
) -> Tuple[
    List[int], List[int], List[str], List[np.ndarray], List[float], List[int], List[int], int,
]:
    r_batch, r_node, r_subtype, r_readout, r_weight, r_t0, r_pd_nd = (
        [], [], [], [], [], [], [],
    )
    skipped_orthogonal = 0
    i_baseline = resolve_i_baseline(i_baseline)
    for b, spec in enumerate(specs):
        t0_by_hex: Dict[int, int] = {}
        if waveform_mse:
            for hex_idx in cost_hex_idxs:
                t_first_sti = hex_first_stim_t(
                    hex_current[b, :, hex_idx], i_baseline=i_baseline,
                )
                t0 = t_first_sti - before_t
                if t0 < 0 or t_first_sti + after_t > n_t:
                    raise ValueError(
                        f"cost window out of range for hex index {hex_idx} "
                        f"spec={spec.name}: t_first_sti={t_first_sti}, n_t={n_t}"
                    )
                t0_by_hex[hex_idx] = t0
        for hex_idx in cost_hex_idxs:
            for subtype in present:
                pref = motion_preference(side, subtype, spec.direction, spec.contrast)
                if pref is None:
                    skipped_orthogonal += 1
                    continue
                nodes = nodes_for_hex_type(b, hex_idx, subtype)
                if len(nodes) == 0:
                    continue
                gt_trace = None
                if waveform_mse:
                    if fig1 is None:
                        raise ValueError("fig1 traces required when waveform_mse=True")
                    trace_id = fig1_key_for_stimulus(side, subtype, spec)
                    if trace_id not in fig1:
                        raise KeyError(f"fig1 trace missing: {trace_id}")
                    gt_trace = fig1[trace_id]
                pd_nd_idx = _pd_nd_index(pref.pd_nd)
                t0 = t0_by_hex.get(hex_idx, 0)
                for uidx in nodes:
                    r_batch.append(b)
                    r_node.append(int(uidx))
                    r_subtype.append(str(subtype))
                    if gt_trace is not None:
                        r_readout.append(gt_trace)
                    r_weight.append(1.0)
                    if waveform_mse:
                        r_t0.append(t0)
                        r_pd_nd.append(pd_nd_idx)
    return (
        r_batch, r_node, r_subtype, r_readout, r_weight, r_t0, r_pd_nd,
        skipped_orthogonal,
    )


def _pack_moving_bar_readout_tensors(
    r_batch, r_node, r_readout, r_weight, r_t0, r_pd_nd, *, device, sim_dtype,
    waveform_mse: bool = True,
):
    n = len(r_batch)
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_node = torch.tensor(np.asarray(r_node), dtype=torch.long, device=device)
    if not waveform_mse:
        gt = torch.zeros((n, 0), dtype=sim_dtype, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return gt, cost_weight, readout_batch, readout_node, None, None, power
    cost_pd_nd = torch.tensor(np.asarray(r_pd_nd), dtype=torch.long, device=device)
    gt = torch.tensor(np.asarray(r_readout), dtype=sim_dtype, device=device)
    cost_t0 = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)
    power = torch.sum(cost_weight[:, None] * gt ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return gt, cost_weight, readout_batch, readout_node, cost_t0, cost_pd_nd, power


def _moving_bar_peak_kwargs(
    contrasts: Sequence[str],
    *,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
) -> dict:
    """Peak-current kwargs for specs that only include the relevant contrasts."""
    contrast_set = frozenset(contrasts)
    kw = {}
    if "bright" in contrast_set and i_bright_moving_bar is not None:
        kw["i_bright_moving_bar"] = float(i_bright_moving_bar)
    if "dark" in contrast_set and i_dark_moving_bar is not None:
        kw["i_dark_moving_bar"] = float(i_dark_moving_bar)
    return kw


def build_moving_bar_gt(
    C,
    device: Optional[str] = None,
    t_onset: int = None,
    *,
    delta_ms: float,
    fig1_path: Path = FIG1_CI_NPZ,
    use_cache: bool = True,
    bar_extent: int = DEFAULT_BAR_EXTENT,
    multi_bar: bool = True,
    cost_extent: Optional[int] = None,
    i_baseline_moving_bar: Optional[float] = None,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
    gt_cells: Optional[Sequence[str]] = None,
    sim_dtype: torch.dtype,
    waveform_mse: bool = True,
) -> MovingBarGt:
    """Build multi-bar stimulus + T4/T5 cost readouts."""
    device = device or C.device
    side = normalize_side(C.meta.get("side", "right"))

    specs = gruntman_moving_bar_specs(contrasts=tuple(contrasts))
    contrast_set = frozenset(contrasts)
    i_baseline_val = resolve_i_baseline(i_baseline_moving_bar)
    stim = build_moving_bar_signals(
        C, specs=specs, t_onset=t_onset, delta_ms=delta_ms,
        bar_extent=bar_extent, multi_bar=bool(multi_bar),
        device=device, use_cache=use_cache,
        network_json=getattr(C, "source_json", None),
        i_baseline=i_baseline_val,
        sim_dtype=sim_dtype,
        **_moving_bar_peak_kwargs(
            contrast_set, i_bright_moving_bar=i_bright_moving_bar, i_dark_moving_bar=i_dark_moving_bar,
        ),
    )
    n_t = int(stim.info["n_t"])
    fig1 = load_fig1_traces(fig1_path, delta_ms=delta_ms) if waveform_mse else None
    before_t = ms_to_t(COST_ALIGNED_FIRST_STI_MS, delta_ms=delta_ms)
    after_t = ms_to_t(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)
    win_t = ms_to_t(COST_WINDOW_MS, delta_ms=delta_ms) + 1

    present = present_gt_cells(
        gt_cells, GT_CELLS, C.cell_names, context="moving_bar",
    )

    cell_names = node_cell_names(C)
    all_sti = sti_hexes(C)
    uv_to_hex_idx = {(int(c.u), int(c.v)): j for j, c in enumerate(all_sti)}
    hexes = moving_bar_cost_hexes(C, cost_extent=cost_extent)
    center_hex = hexes[0] if cost_extent == 0 and len(hexes) == 1 else None
    cost_hex_idxs = [uv_to_hex_idx[(int(c.u), int(c.v))] for c in hexes]
    hex_by_idx = {idx: c for c, idx in zip(hexes, cost_hex_idxs)}

    def _nodes_for_hex_type(_b, hex_idx, subtype):
        hx = hex_by_idx[hex_idx]
        return col2gt(C, hx.u, hx.v, subtype, cell_names)

    rows = _assemble_moving_bar_readouts(
        specs=stim.specs,
        hex_current=stim.hex_current,
        cost_hex_idxs=cost_hex_idxs,
        i_baseline=i_baseline_val,
        before_t=before_t,
        after_t=after_t,
        n_t=n_t,
        side=side,
        fig1=fig1,
        present=present,
        nodes_for_hex_type=_nodes_for_hex_type,
        waveform_mse=waveform_mse,
    )
    (
        r_batch, r_node, r_subtype, r_readout, r_weight, r_t0, r_pd_nd,
        skipped_orthogonal,
    ) = rows

    if not r_batch:
        raise ValueError("no moving-bar cost nodes (check subtypes and sti hexes)")

    gt, cost_weight, readout_batch, readout_node, cost_t0, cost_pd_nd, power = (
        _pack_moving_bar_readout_tensors(
            r_batch, r_node, r_readout, r_weight, r_t0, r_pd_nd,
            device=device, sim_dtype=sim_dtype, waveform_mse=waveform_mse,
        )
    )
    (
        dsi_pos_rows, dsi_neg_rows, dsi_pos_ptr, dsi_neg_ptr,
        dsi_tgt, dsi_w, dsi_pow,
    ) = build_dsi_pack_fields(
        stim.specs, r_batch, r_subtype, r_weight,
        side=side, device=device, sim_dtype=sim_dtype,
    )

    info = {
        **stim.info,
        "n_cost": int(readout_batch.shape[0]),
        "n_cost_pd": int((cost_pd_nd == PD_IDX).sum().item()) if cost_pd_nd is not None else 0,
        "n_cost_nd": int((cost_pd_nd == ND_IDX).sum().item()) if cost_pd_nd is not None else 0,
        "n_cost_dsi": int(dsi_tgt.shape[0]),
        "n_batch": stim.info["n_batch"],
        "n_cost_hexes": len(hexes),
        "cost_extent": cost_extent,
        "cost_hex_uv": (int(center_hex.u), int(center_hex.v)) if center_hex else None,
        "side": side,
        "present_gts": present,
        "skipped_orthogonal": skipped_orthogonal,
        "waveform_mse": bool(waveform_mse),
        "delta_ms": float(delta_ms),
    }
    if waveform_mse:
        info.update({
            "fig1_path": str(fig1_path),
            "cost_window_before_ms": COST_WINDOW_BEFORE_MS,
            "cost_window_after_ms": COST_WINDOW_AFTER_MS,
            "cost_window_ms": COST_WINDOW_MS,
            "cost_aligned_first_sti_ms": COST_ALIGNED_FIRST_STI_MS,
            "cost_window_t": win_t,
        })
    return MovingBarGt(
        i_sti=stim.i_sti,
        gt=gt,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_node=readout_node,
        cost_pd_nd=cost_pd_nd,
        n_batch=stim.info["n_batch"],
        n_t=n_t,
        info=info,
        dsi_pos_rows=dsi_pos_rows,
        dsi_neg_rows=dsi_neg_rows,
        dsi_pos_ptr=dsi_pos_ptr,
        dsi_neg_ptr=dsi_neg_ptr,
        dsi_gt=dsi_tgt,
        dsi_weight=dsi_w,
        dsi_power=dsi_pow,
    )


# -- Session-level t0 grids + readout specs (need session + side) -------------


@dataclass
class MovingBarSessionT0:
    t0_bn: np.ndarray
    before_t: Dict[str, int]
    after_t: Dict[str, int]
    side: str
    n_filter_hexes: int


def bar_specs_for_session(session, task) -> List[MovingBarSpec]:
    """Gruntman bar specs for ``task`` (bright/dark)."""
    if session.backend.network is None:
        raise ValueError("bar_specs_for_session requires session.backend.network")
    contrast = "bright" if "bright" in task else "dark"
    return list(gruntman_moving_bar_specs(contrasts=(contrast,)))


def moving_bar_session_t0_grids(
    session,
    specs: Sequence[MovingBarSpec],
    cost_extent,
    n_t: int,
    *,
    at_x=None,
    at_y=None,
    t_onset: int = None,
    delta_ms: float,
) -> MovingBarSessionT0:
    """Session-level ``t0`` / horizon grids for moving-bar cost or analyze."""
    C = session.backend.network
    if C is None:
        raise ValueError("moving_bar_session_t0_grids requires session.backend.network")
    i_baseline = moving_bar_i_baseline_from_opts(session.train_opts)

    side = normalize_side(C.meta.get('side', 'right'))
    all_hexes = moving_bar_cost_hexes(C, cost_extent=cost_extent)
    if at_x is not None or at_y is not None:
        filt_hexes = filter_sti_hexes(all_hexes, at_x=at_x, at_y=at_y)
        if not filt_hexes:
            raise SystemExit(
                f'no sti hexes match x={at_x!r} y={at_y!r} within cost_extent',
            )
    else:
        filt_hexes = all_hexes
    stim = build_moving_bar_signals(
        C, specs=specs, n_t=n_t, t_onset=t_onset, delta_ms=delta_ms,
        device=C.node_cell.device, i_baseline=i_baseline,
    )
    uv_to_idx = {(int(col.u), int(col.v)): j for j, col in enumerate(sti_hexes(C))}
    all_hex_idxs = [uv_to_idx[(int(c.u), int(c.v))] for c in all_hexes]
    filt_hex_idxs = [uv_to_idx[(int(c.u), int(c.v))] for c in filt_hexes]
    grids = build_moving_bar_t0_grids(
        stim.hex_current, specs, n_t, i_baseline,
        all_hex_idxs=all_hex_idxs,
        filt_hex_idxs=filt_hex_idxs,
        network_C=C,
        filt_network_hexes=filt_hexes,
    )
    return MovingBarSessionT0(
        t0_bn=grids.t0_bn,
        before_t=grids.before_t,
        after_t=grids.after_t,
        side=side,
        n_filter_hexes=len(filt_hexes),
    )


def _pack_readout_cell_names(session, task: str) -> List[str]:
    """Unique cell names on ``pack.readout_node`` (pack order)."""
    pack = session.pack_for(task)
    u = pack.readout_node
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    u = np.asarray(u, dtype=np.int64)
    C = session.backend.network
    if C is None:
        raise ValueError("_pack_readout_cell_names requires session.backend.network")
    node_cell = C.node_cell[u]
    if torch.is_tensor(node_cell):
        node_cell = node_cell.detach().cpu().numpy()
    names = list(C.cell_names)
    seq = [str(names[int(ti)]) for ti in node_cell]
    seen: set = set()
    out: List[str] = []
    for name in seq:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def moving_bar_row_specs(session, task: str, side: str) -> Dict[str, List[str]]:
    """Per-readout-cell active bar spec names for ``side`` and task contrast."""
    contrast = "bright" if "bright" in task else "dark"
    return {
        st: [
            f'{d}_{c}_{w}'
            for d, c, w in active_stimuli_for_subtype(side, st)
            if c == contrast
        ]
        for st in _pack_readout_cell_names(session, task)
    }


# -- Moving-bar stimulus_opts builders (was FiveCol) --------------------------


def make_moving_bar_stimulus_opts(
    polarity: str,
    *,
    i_baseline_moving_bar: float,
    i_moving_bar: float,
    pre_ms: float,
    delta_ms: float,
    multi_bar: bool,
    gt_cells=None,
):
    """PR moving-bar stimulus opts for ``moving_bar_{polarity}``."""
    if polarity not in MOVING_BAR_POLARITIES:
        raise ValueError(f"moving-bar polarity must be 'bright' or 'dark', got {polarity!r}")
    bar_key = "i_bright_moving_bar" if polarity == "bright" else "i_dark_moving_bar"
    out = {
        "i_baseline_moving_bar": float(i_baseline_moving_bar),
        bar_key: float(i_moving_bar),
        "pre_ms": float(pre_ms),
        "delta_ms": float(delta_ms),
        "multi_bar": bool(multi_bar),
    }
    rs = normalize_gt_cells(gt_cells)
    if rs is not None:
        out["gt_cells"] = rs
    return out


def session_moving_bar_i_baseline(train_opts) -> float:
    """``i_baseline_moving_bar`` from moving-bar stimulus opts on a train session."""
    return moving_bar_i_baseline_from_opts(train_opts)


def _enrich_moving_bar_stimulus_opts(opts, info, *, cost_extent):
    """Attach runtime fields from a built moving-bar task; keep canonical ``i_*``."""
    out = dict(opts)
    out["n_t"] = int(info["n_t"])
    if out.get("delta_ms") is None:
        raise ValueError("moving-bar stimulus opts require delta_ms")
    dt = float(out["delta_ms"])
    out["pre_ms"] = float(info["t_onset"]) * dt
    out["spec_names"] = list(info["spec_names"])
    if cost_extent is not None:
        out["cost_extent"] = int(cost_extent)
    out["delta_ms"] = dt
    if "present_gts" in info:
        out["gt_cells"] = list(info["present_gts"])
    return out
