# -*- coding: utf-8 -*-
"""Direction selectivity index (DSI) for moving-bar plots and training.

Owns axis DSI math, cost-row pairing, pack tensors, remap, and cost-from-sel.
Core callers (:mod:`network.moving_bar_target`, :mod:`FiveCol_MedSim_Pytorch`)
and plot (:mod:`plot.moving_bar`) import from here.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

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
    r_unit: Sequence[int],
    r_target: Sequence[np.ndarray],
    r_weight: Sequence[float],
) -> Tuple[List[int], List[int], List[float], List[float]]:
    """Pair cost rows on each axis (pos=right|up, neg=left|down).

    Returns ``(row_pos, row_neg, target_dsi, weight)``.
    """
    row_by_bu: dict[tuple[int, int], int] = {}
    for i, (b, u) in enumerate(zip(r_batch, r_unit)):
        row_by_bu[(int(b), int(u))] = i

    batches_by_dir_w: dict[tuple[str, str], list[int]] = {}
    for bi, spec in enumerate(specs):
        key = (spec.direction, "w1" if float(spec.width_deg) <= 3.0 else "w4")
        batches_by_dir_w.setdefault(key, []).append(bi)

    row_pos, row_neg, targets, weights = [], [], [], []
    units = sorted({int(u) for u in r_unit})
    for pos_dir, neg_dir in AXIS_DIRECTION_PAIRS:
        wtags = {
            w for (d, w) in batches_by_dir_w if d in (pos_dir, neg_dir)
        }
        for wtag in sorted(wtags):
            pos_batches = batches_by_dir_w.get((pos_dir, wtag), [])
            neg_batches = batches_by_dir_w.get((neg_dir, wtag), [])
            if not pos_batches or not neg_batches:
                continue
            for pb in pos_batches:
                for nb in neg_batches:
                    for unit in units:
                        ip = row_by_bu.get((pb, unit))
                        inn = row_by_bu.get((nb, unit))
                        if ip is None or inn is None:
                            continue
                        dsi = axis_dsi(
                            float(np.max(r_target[ip])),
                            float(np.max(r_target[inn])),
                        )
                        if dsi is None:
                            continue
                        row_pos.append(ip)
                        row_neg.append(inn)
                        targets.append(float(dsi))
                        weights.append(0.5 * (float(r_weight[ip]) + float(r_weight[inn])))
    return row_pos, row_neg, targets, weights


def pack_moving_bar_dsi_tensors(
    row_pos, row_neg, targets, weights, *, device, sim_dtype,
):
    if not row_pos:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros(0, dtype=sim_dtype, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return empty_long, empty_long, empty, empty, power
    dsi_row_pos = torch.tensor(np.asarray(row_pos), dtype=torch.long, device=device)
    dsi_row_neg = torch.tensor(np.asarray(row_neg), dtype=torch.long, device=device)
    dsi_target = torch.tensor(np.asarray(targets), dtype=sim_dtype, device=device)
    dsi_weight = torch.tensor(np.asarray(weights), dtype=sim_dtype, device=device)
    power = torch.sum(dsi_weight * dsi_target ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return dsi_row_pos, dsi_row_neg, dsi_target, dsi_weight, power


def build_dsi_pack_fields(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_unit: Sequence[int],
    r_target: Sequence[np.ndarray],
    r_weight: Sequence[float],
    *,
    device,
    sim_dtype,
):
    """Assemble + tensorize DSI fields for a moving-bar target build."""
    return pack_moving_bar_dsi_tensors(
        *assemble_moving_bar_dsi_pairs(specs, r_batch, r_unit, r_target, r_weight),
        device=device,
        sim_dtype=sim_dtype,
    )


def remap_dsi_rows(pack, kept_old_rows) -> dict:
    """Remap ``pack.dsi_*`` onto kept cost-row indices; drop incomplete pairs."""
    if pack.dsi_row_pos is None or pack.dsi_row_pos.numel() == 0:
        return {
            "dsi_row_pos": pack.dsi_row_pos,
            "dsi_row_neg": pack.dsi_row_neg,
            "dsi_target": pack.dsi_target,
            "dsi_weight": pack.dsi_weight,
            "dsi_power": pack.dsi_power,
        }
    device = pack.dsi_row_pos.device
    n = int(pack.data.shape[0])
    kept = torch.as_tensor(kept_old_rows, dtype=torch.long, device=device)
    lut = torch.full((n,), -1, dtype=torch.long, device=device)
    lut[kept] = torch.arange(kept.numel(), dtype=torch.long, device=device)
    new_pos = lut[pack.dsi_row_pos]
    new_neg = lut[pack.dsi_row_neg]
    ok = (new_pos >= 0) & (new_neg >= 0)
    if not bool(ok.any()):
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros(0, dtype=pack.dsi_target.dtype, device=device)
        power = torch.tensor(1.0, dtype=pack.dsi_power.dtype, device=device)
        return {
            "dsi_row_pos": empty_long,
            "dsi_row_neg": empty_long,
            "dsi_target": empty,
            "dsi_weight": empty,
            "dsi_power": power,
        }
    dsi_target = pack.dsi_target[ok]
    dsi_weight = pack.dsi_weight[ok]
    power = torch.sum(dsi_weight * dsi_target ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=dsi_target.dtype, device=device)
    return {
        "dsi_row_pos": new_pos[ok],
        "dsi_row_neg": new_neg[ok],
        "dsi_target": dsi_target,
        "dsi_weight": dsi_weight,
        "dsi_power": power,
    }


def cost_dsi_from_sel(pack, scale: torch.Tensor, sel: torch.Tensor) -> Optional[torch.Tensor]:
    """Unweighted DSI MSE (% of dsi_power); None if no complete pairs."""
    if pack.dsi_row_pos is None or pack.dsi_row_pos.numel() == 0:
        return None
    pos = pack.dsi_row_pos
    neg = pack.dsi_row_neg
    model_pos = (scale[pos, None] * sel[pos]).amax(dim=-1)
    model_neg = (scale[neg, None] * sel[neg]).amax(dim=-1)
    model_dsi = axis_dsi_torch(model_pos, model_neg)
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
