# -*- coding: utf-8 -*-
"""T4 / T5 moving-bar preference + DSI.

Rules match ``t4_t5_dsi.md``. Orthogonal motion (``—`` in the tables)
returns ``None`` so those (stimulus, subtype) pairs are skipped in training.

Also owns DSI math, cost-row pairing, pack tensors, remap, and cost-from-sel
for core (:mod:`network.moving_bar_target`, :mod:`FiveCol_MedSim_Pytorch`)
and plot (:mod:`plot.moving_bar`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from visual_stimulus.moving_bar_stimulus import GRUNTMAN_WIDTHS_DEG, MovingBarSpec

READOUT_SUBTYPES: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

READOUT_SUBTYPE_ALIASES: dict = {
    "T4": tuple(st for st in READOUT_SUBTYPES if st.startswith("T4")),
    "T5": tuple(st for st in READOUT_SUBTYPES if st.startswith("T5")),
}

_HORIZONTAL = frozenset({"right", "left"})
_VERTICAL = frozenset({"up", "down"})
# Plot / stimulus iteration order (matches gruntman_moving_bar_specs).
_HORIZONTAL_AXIS: Tuple[str, ...] = ("right", "left")
_VERTICAL_AXIS: Tuple[str, ...] = ("up", "down")
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}

# Subtype PD on the **right** eye (anterior->posterior = right; c/d = up/down).
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


def dsi_sequential_batch_pairs(spec_names: Sequence[str]) -> Tuple[Tuple[int, ...], ...]:
    """Minimal stimulus-batch groups for sequential DSI: one ``(pos,neg)`` per axis×width.

    Spec names are ``direction_contrast_wtag`` (e.g. ``right_bright_w1``). Each returned
    tuple holds the batch indices that must share one forward to form complete DSI pairs.
    """
    batches_by_dir_w: dict[tuple[str, str], list[int]] = {}
    for bi, sname in enumerate(spec_names):
        direction, _contrast, wtag = parse_moving_bar_spec(sname)
        batches_by_dir_w.setdefault((direction, wtag), []).append(int(bi))
    pairs: list[tuple[int, ...]] = []
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
            pairs.append(tuple(sorted({*pos, *neg})))
    return tuple(pairs)

# Hardcoded |DSI| = |(PD−ND)/(PD+ND)| from ``t4_t5_dsi.md`` source table.
# Keys: ``{T4|T5}_{PC|NC}_{w1|w4}``. Sign applied at assemble time from axis PD/ND.
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

    pd_nd: str  # "PD" | "ND"
    pc_nc: str  # "PC" | "NC"


def expand_remove_subtypes_list(names: Sequence[str]) -> Tuple[str, ...]:
    """Expand ``remove_moving_bar`` SUBTYPES ``READOUT_SUBTYPE_ALIASES`` (e.g. T4, T5)."""
    if not names:
        raise ValueError("readout_subtypes must not be empty")
    out: list = []
    seen: set = set()
    for raw in names:
        key = str(raw).strip()
        if key in READOUT_SUBTYPE_ALIASES:
            pool = READOUT_SUBTYPE_ALIASES[key]
        elif key in READOUT_SUBTYPES:
            pool = (key,)
        else:
            valid = ", ".join((*READOUT_SUBTYPE_ALIASES, *READOUT_SUBTYPES))
            raise ValueError(f"unknown readout subtype {key!r} (expected {valid})")
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
    """Map one cardinal stimulus to PD/ND + PC/NC for a T4/T5 subtype.

    Returns ``None`` when motion is orthogonal to the subtype PD axis (table ``—``).
    """
    if subtype not in READOUT_SUBTYPES:
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
    """Non-orthogonal (direction, contrast, width_tag) triples for one subtype (8 total)."""
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
    """Elementwise axis DSI; denom ≤ 0 → 0."""
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


def assemble_moving_bar_dsi_pairs(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_subtype: Sequence[str],
    r_weight: Sequence[float],
    *,
    side: str,
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """One DSI group per ``(subtype, contrast, wtag, axis)``.

    Returns ``(pos_row_groups, neg_row_groups, target_dsi, weight)``.
    Target DSI is the hardcoded ``FIG1_ABS_DSI`` table (signed for axis pos).
    """
    batches_by_condition: dict[tuple[str, str, str], list[int]] = {}
    for bi, spec in enumerate(specs):
        key = (spec.direction, spec.contrast, width_tag(spec.width_deg))
        batches_by_condition.setdefault(key, []).append(bi)

    rows_by_subtype_batch: dict[tuple[str, int], list[int]] = {}
    for i, (b, st) in enumerate(zip(r_batch, r_subtype)):
        rows_by_subtype_batch.setdefault((str(st), int(b)), []).append(i)

    pos_groups: List[List[int]] = []
    neg_groups: List[List[int]] = []
    targets: List[float] = []
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
                targets.append(float(dsi))
                weights.append(0.5 * (w_pos + w_neg))
    return pos_groups, neg_groups, targets, weights


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


def pack_moving_bar_dsi_tensors(
    pos_groups, neg_groups, targets, weights, *, device, sim_dtype,
):
    if not pos_groups:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros(0, dtype=sim_dtype, device=device)
        ptr0 = torch.zeros(1, dtype=torch.long, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return empty_long, empty_long, ptr0, ptr0, empty, empty, power
    dsi_pos_rows, dsi_pos_ptr = _csr_from_groups(pos_groups, device=device)
    dsi_neg_rows, dsi_neg_ptr = _csr_from_groups(neg_groups, device=device)
    dsi_target = torch.tensor(np.asarray(targets), dtype=sim_dtype, device=device)
    dsi_weight = torch.tensor(np.asarray(weights), dtype=sim_dtype, device=device)
    power = torch.sum(dsi_weight * dsi_target ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return (
        dsi_pos_rows, dsi_neg_rows, dsi_pos_ptr, dsi_neg_ptr,
        dsi_target, dsi_weight, power,
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
    """Assemble + tensorize subtype-grouped DSI fields for a moving-bar target."""
    return pack_moving_bar_dsi_tensors(
        *assemble_moving_bar_dsi_pairs(
            specs, r_batch, r_subtype, r_weight, side=side,
        ),
        device=device,
        sim_dtype=sim_dtype,
    )


def _empty_dsi_fields(pack, device) -> dict:
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    empty = torch.zeros(0, dtype=pack.dsi_target.dtype, device=device)
    ptr0 = torch.zeros(1, dtype=torch.long, device=device)
    power = torch.tensor(1.0, dtype=pack.dsi_power.dtype, device=device)
    return {
        "dsi_pos_rows": empty_long,
        "dsi_neg_rows": empty_long,
        "dsi_pos_ptr": ptr0,
        "dsi_neg_ptr": ptr0,
        "dsi_target": empty,
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
            "dsi_target": pack.dsi_target,
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
    dsi_target = pack.dsi_target[ix]
    dsi_weight = pack.dsi_weight[ix]
    power = torch.sum(dsi_weight * dsi_target ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=dsi_target.dtype, device=device)
    return {
        "dsi_pos_rows": dsi_pos_rows,
        "dsi_neg_rows": dsi_neg_rows,
        "dsi_pos_ptr": dsi_pos_ptr,
        "dsi_neg_ptr": dsi_neg_ptr,
        "dsi_target": dsi_target,
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
    """Unweighted DSI MSE (% of dsi_power); None if no complete groups.

    Model peak per side = mean over subtype unit rows of ``amax(scale * sel)``.
    """
    if pack.dsi_pos_ptr is None or int(pack.dsi_pos_ptr.numel()) <= 1:
        return None
    peak_pos_u = (scale[pack.dsi_pos_rows, None] * sel[pack.dsi_pos_rows]).amax(dim=-1)
    peak_neg_u = (scale[pack.dsi_neg_rows, None] * sel[pack.dsi_neg_rows]).amax(dim=-1)
    if not (torch.isfinite(peak_pos_u).all() and torch.isfinite(peak_neg_u).all()):
        raise RuntimeError("non-finite DSI peaks (NaN/Inf in readout)")
    peak_pos = _csr_group_mean(peak_pos_u, pack.dsi_pos_ptr)
    peak_neg = _csr_group_mean(peak_neg_u, pack.dsi_neg_ptr)
    model_dsi = axis_dsi_torch(peak_pos, peak_neg)
    diff = model_dsi - pack.dsi_target
    return torch.sum(pack.dsi_weight * diff ** 2) / pack.dsi_power * 100.0


def moving_bar_dsi_for_spec(
    trace_map: Mapping[tuple, np.ndarray],
    cell_name: str,
    spec_name: str,
) -> Optional[float]:
    """DSI for one cell×spec: (this dir − opposite) / (this + opposite)."""
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
    model_dsi: Optional[float] = None,
    data_dsi: Optional[float] = None,
    *,
    has_data: bool = False,
) -> str:
    """Append DSI lines to a subplot title *head* (e.g. from ``cell_title_with_n``)."""
    lines = [str(head)]
    if model_dsi is not None:
        lines.append(f"DSI={model_dsi:.3f}")
    if has_data and data_dsi is not None:
        lines.append(f"data DSI={data_dsi:.3f}")
    return "\n".join(lines)
