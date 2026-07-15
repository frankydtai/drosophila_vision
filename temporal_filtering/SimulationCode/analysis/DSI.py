# -*- coding: utf-8 -*-
"""Direction selectivity index (DSI) for moving-bar plots and training.

Owns axis DSI math, cost-row pairing, pack tensors, remap, and cost-from-sel.
Core callers (:mod:`network.moving_bar_target`, :mod:`FiveCol_MedSim_Pytorch`)
and plot (:mod:`plot.moving_bar`) import from here.

Training DSI is one term per ``(subtype, width, axis)``: fig1 target once,
model peak = mean over that subtype's unit cost-rows, then axis DSI.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from visual_stimulus.moving_bar_stimulus import MovingBarSpec

AXIS_DIRECTION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("right", "left"),
    ("up", "down"),
)
_POS_DIRS = frozenset(pos for pos, _neg in AXIS_DIRECTION_PAIRS)
_DIR_TO_AXIS = {
    **{pos: (pos, neg) for pos, neg in AXIS_DIRECTION_PAIRS},
    **{neg: (pos, neg) for pos, neg in AXIS_DIRECTION_PAIRS},
}


def parse_moving_bar_spec(sname: str) -> Tuple[str, str, str]:
    direction, contrast, wtag = str(sname).split("_", 2)
    return direction, contrast, wtag


def width_tag_from_deg(width_deg: float) -> str:
    return "w1" if float(width_deg) <= 3.0 else "w4"


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


def assemble_moving_bar_dsi_pairs(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_subtype: Sequence[str],
    r_target: Sequence[np.ndarray],
    r_weight: Sequence[float],
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """One DSI group per ``(subtype, wtag, axis)``.

    Returns ``(pos_row_groups, neg_row_groups, target_dsi, weight)``.
    """
    batches_by_dir_w: dict[tuple[str, str], list[int]] = {}
    for bi, spec in enumerate(specs):
        key = (spec.direction, width_tag_from_deg(spec.width_deg))
        batches_by_dir_w.setdefault(key, []).append(bi)

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
            wtags = {
                w for (d, w) in batches_by_dir_w if d in (pos_dir, neg_dir)
            }
            for wtag in sorted(wtags):
                pos_batches = batches_by_dir_w.get((pos_dir, wtag), [])
                neg_batches = batches_by_dir_w.get((neg_dir, wtag), [])
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
                dsi = axis_dsi(
                    float(np.max(r_target[pos_rows[0]])),
                    float(np.max(r_target[neg_rows[0]])),
                )
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
    r_target: Sequence[np.ndarray],
    r_weight: Sequence[float],
    *,
    device,
    sim_dtype,
):
    """Assemble + tensorize subtype-grouped DSI fields for a moving-bar target."""
    return pack_moving_bar_dsi_tensors(
        *assemble_moving_bar_dsi_pairs(specs, r_batch, r_subtype, r_target, r_weight),
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
    n = int(pack.data.shape[0])
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
    label: str,
    n: Optional[int] = None,
    model_dsi: Optional[float] = None,
    data_dsi: Optional[float] = None,
    *,
    has_data: bool = False,
) -> str:
    if n is None:
        lines = [str(label)]
    else:
        lines = [f"{label} (n={int(n)})"]
    if model_dsi is not None:
        lines.append(f"DSI={model_dsi:.3f}")
    if has_data and data_dsi is not None:
        lines.append(f"data DSI={data_dsi:.3f}")
    return "\n".join(lines)
