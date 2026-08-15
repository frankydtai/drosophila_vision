# -*- coding: utf-8 -*-
"""Moving-bar paradigm pack: bind GT numbers to the network for cost.

Cost hexes, sti ``i_sti``, :class:`MovingBarGt` packing, and DSI entry CSR.
GT traces and motion preference come from :mod:`task.moving_bar.gt`.
``Pack`` wrapping lives in the ``train`` layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neuron.param import t_from_ms
from network.construction import (
    hex2gt,
    active_gt_cells,
    normalize_gt_cells,
    node_cells,
)
from task.moving_bar.gt import (
    FIG1_CI_NPZ,
    GT_CELLS,
    active_stis_for_subtype,
    axis_dsi_torch,
    fig1_trace_for_sti,
    hardcoded_axis_dsi,
    load_fig1_traces,
    motion_preference,
    normalize_side,
    width_tag,
)
from task.moving_bar.sti_geo import (
    BAR_RADIUS,
    _as_int64_np,
    filter_sti_hexes,
    moving_bar_cost_hexes,
    network_uv_np,
    sti_hexes,
)
from task.moving_bar.sti_spec import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_AFTER_MS,
    COST_WINDOW_BEFORE_MS,
    COST_WINDOW_MS,
    MovingBarSpec,
    ND_IDX,
    PD_IDX,
    build_moving_bar_signals,
    build_moving_bar_t0_grids,
    gruntman_moving_bar_specs,
    hex_first_sti_t,
    resolve_moving_bar_i_baseline,
    resolve_i_baseline,
)

MOVING_BAR_CONTRASTS = frozenset({"bright", "dark"})


@dataclass
class MovingBarGt:
    i_sti: torch.Tensor
    gts: torch.Tensor
    power: torch.Tensor
    cost_scales: torch.Tensor
    entry_batches: torch.Tensor
    entry_nodes: torch.Tensor
    n_batch: int
    n_t: int
    info: dict
    cost_t0s: Optional[torch.Tensor] = None
    cost_pd_nds: Optional[torch.Tensor] = None
    dsi_pos_entries: Optional[torch.Tensor] = None
    dsi_neg_entries: Optional[torch.Tensor] = None
    dsi_pos_ptr: Optional[torch.Tensor] = None
    dsi_neg_ptr: Optional[torch.Tensor] = None
    dsi_gts: Optional[torch.Tensor] = None
    dsi_scales: Optional[torch.Tensor] = None
    dsi_power: Optional[torch.Tensor] = None


def moving_bar_nodes_on_hexes(connectome, cell: str, hexes: Sequence) -> np.ndarray:
    """Node indices of ``cell`` on any of ``hexes`` (vectorized axial uv pack)."""
    if not hexes:
        return np.zeros(0, dtype=np.int64)
    if cell not in connectome.cells:
        raise ValueError(f"unknown cell {cell!r}; known: {list(connectome.cells)}")
    ti = int(connectome.cells.index(cell))
    node_u_np, node_v_np = network_uv_np(connectome)
    cell_ids = _as_int64_np(connectome.node_cells)
    uv_span = int(max(np.max(np.abs(node_u_np)), np.max(np.abs(node_v_np)), 1)) + 1
    pack = (node_u_np + uv_span) * (2 * uv_span + 1) + (node_v_np + uv_span)
    hex_pack = np.array(
        [
            (int(hex.u) + uv_span) * (2 * uv_span + 1) + (int(hex.v) + uv_span)
            for hex in hexes
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


def assemble_moving_bar_dsi_groups(
    specs: Sequence[MovingBarSpec],
    r_batch: Sequence[int],
    r_subtype: Sequence[str],
    r_scale: Sequence[float],
    *,
    side: str,
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """One DSI group per ``(subtype, contrast, wtag, axis)``."""
    batches_by_condition: dict[tuple[str, str, str], list[int]] = {}
    for bi, spec in enumerate(specs):
        key = (spec.direction, spec.contrast, width_tag(spec.width_deg))
        batches_by_condition.setdefault(key, []).append(bi)

    entries_by_subtype_batch: dict[tuple[str, int], list[int]] = {}
    for i, (b, st) in enumerate(zip(r_batch, r_subtype)):
        entries_by_subtype_batch.setdefault((str(st), int(b)), []).append(i)

    pos_groups: List[List[int]] = []
    neg_groups: List[List[int]] = []
    dsi_vals: List[float] = []
    scales: List[float] = []
    subtypes = sorted({str(st) for st in r_subtype})
    for subtype in subtypes:
        for pos_dir, neg_dir in (("right", "left"), ("up", "down")):
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
                pos_entries: list[int] = []
                for pb in pos_batches:
                    pos_entries.extend(entries_by_subtype_batch.get((subtype, pb), []))
                neg_entries: list[int] = []
                for nb in neg_batches:
                    neg_entries.extend(entries_by_subtype_batch.get((subtype, nb), []))
                if not pos_entries or not neg_entries:
                    continue
                dsi = hardcoded_axis_dsi(side, subtype, specs[pos_batches[0]])
                if dsi is None:
                    continue
                w_pos = float(np.mean([float(r_scale[i]) for i in pos_entries]))
                w_neg = float(np.mean([float(r_scale[i]) for i in neg_entries]))
                pos_groups.append(pos_entries)
                neg_groups.append(neg_entries)
                dsi_vals.append(float(dsi))
                scales.append(0.5 * (w_pos + w_neg))
    return pos_groups, neg_groups, dsi_vals, scales


def _csr_from_groups(
    groups: Sequence[Sequence[int]], *, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(flat_entries, ptr)`` with ``ptr`` length ``n_groups + 1``."""
    ptr = [0]
    flat: list[int] = []
    for dsi_group in groups:
        flat.extend(int(entry_i) for entry_i in dsi_group)
        ptr.append(len(flat))
    entries_t = torch.tensor(np.asarray(flat, dtype=np.int64), dtype=torch.long, device=device)
    ptr_t = torch.tensor(np.asarray(ptr, dtype=np.int64), dtype=torch.long, device=device)
    return entries_t, ptr_t


def pack_moving_bar_dsi_tensors(pos_groups, neg_groups, dsi_vals, scales, *, device, sim_dtype):
    if not pos_groups:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        empty = torch.zeros(0, dtype=sim_dtype, device=device)
        ptr0 = torch.zeros(1, dtype=torch.long, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return empty_long, empty_long, ptr0, ptr0, empty, empty, power
    dsi_pos_entries, dsi_pos_ptr = _csr_from_groups(pos_groups, device=device)
    dsi_neg_entries, dsi_neg_ptr = _csr_from_groups(neg_groups, device=device)
    dsi_gts = torch.tensor(np.asarray(dsi_vals), dtype=sim_dtype, device=device)
    dsi_scales = torch.tensor(np.asarray(scales), dtype=sim_dtype, device=device)
    power = torch.sum(dsi_scales * dsi_gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return (
        dsi_pos_entries, dsi_neg_entries, dsi_pos_ptr, dsi_neg_ptr,
        dsi_gts, dsi_scales, power,
    )


def _empty_dsi_fields(pack, device) -> dict:
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    empty = torch.zeros(0, dtype=pack.dsi_gts.dtype, device=device)
    ptr0 = torch.zeros(1, dtype=torch.long, device=device)
    power = torch.tensor(1.0, dtype=pack.dsi_power.dtype, device=device)
    return {
        "dsi_pos_entries": empty_long,
        "dsi_neg_entries": empty_long,
        "dsi_pos_ptr": ptr0,
        "dsi_neg_ptr": ptr0,
        "dsi_gts": empty,
        "dsi_scales": empty,
        "dsi_power": power,
    }


def remap_dsi_entries(pack, kept_old_entries) -> dict:
    """Remap CSR DSI members onto kept cost-entry indices; drop incomplete groups."""
    if pack.dsi_pos_ptr is None or int(pack.dsi_pos_ptr.numel()) <= 1:
        return {
            "dsi_pos_entries": pack.dsi_pos_entries,
            "dsi_neg_entries": pack.dsi_neg_entries,
            "dsi_pos_ptr": pack.dsi_pos_ptr,
            "dsi_neg_ptr": pack.dsi_neg_ptr,
            "dsi_gts": pack.dsi_gts,
            "dsi_scales": pack.dsi_scales,
            "dsi_power": pack.dsi_power,
        }
    device = pack.dsi_pos_entries.device
    n = int(pack.entry_batches.shape[0])
    kept = torch.as_tensor(kept_old_entries, dtype=torch.long, device=device)
    lut = torch.full((n,), -1, dtype=torch.long, device=device)
    lut[kept] = torch.arange(kept.numel(), dtype=torch.long, device=device)

    n_dsi = int(pack.dsi_pos_ptr.numel()) - 1
    new_pos_groups: list[list[int]] = []
    new_neg_groups: list[list[int]] = []
    kept_dsi_group_indices: list[int] = []
    pos_entries = pack.dsi_pos_entries
    neg_entries = pack.dsi_neg_entries
    pos_ptr = pack.dsi_pos_ptr
    neg_ptr = pack.dsi_neg_ptr
    for dsi_group_idx in range(n_dsi):
        p0, p1 = int(pos_ptr[dsi_group_idx]), int(pos_ptr[dsi_group_idx + 1])
        n0, n1 = int(neg_ptr[dsi_group_idx]), int(neg_ptr[dsi_group_idx + 1])
        new_pos = lut[pos_entries[p0:p1]]
        new_neg = lut[neg_entries[n0:n1]]
        new_pos = new_pos[new_pos >= 0]
        new_neg = new_neg[new_neg >= 0]
        if new_pos.numel() == 0 or new_neg.numel() == 0:
            continue
        new_pos_groups.append(new_pos.tolist())
        new_neg_groups.append(new_neg.tolist())
        kept_dsi_group_indices.append(dsi_group_idx)
    if not kept_dsi_group_indices:
        return _empty_dsi_fields(pack, device)
    dsi_pos_entries, dsi_pos_ptr = _csr_from_groups(new_pos_groups, device=device)
    dsi_neg_entries, dsi_neg_ptr = _csr_from_groups(new_neg_groups, device=device)
    kept_dsi_group_idx = torch.tensor(kept_dsi_group_indices, dtype=torch.long, device=device)
    dsi_gts = pack.dsi_gts[kept_dsi_group_idx]
    dsi_scales = pack.dsi_scales[kept_dsi_group_idx]
    power = torch.sum(dsi_scales * dsi_gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=dsi_gts.dtype, device=device)
    return {
        "dsi_pos_entries": dsi_pos_entries,
        "dsi_neg_entries": dsi_neg_entries,
        "dsi_pos_ptr": dsi_pos_ptr,
        "dsi_neg_ptr": dsi_neg_ptr,
        "dsi_gts": dsi_gts,
        "dsi_scales": dsi_scales,
        "dsi_power": power,
    }


def _csr_span_mean(vals: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Mean of ``vals`` over CSR segments defined by ``ptr``."""
    n_g = int(ptr.numel()) - 1
    if n_g == 0:
        return vals.new_zeros((0,))
    n_segment = ptr[1:] - ptr[:-1]
    csr_segment_idx = torch.repeat_interleave(
        torch.arange(n_g, device=vals.device, dtype=torch.long),
        n_segment,
    )
    sums = vals.new_zeros((n_g,))
    sums.scatter_add_(0, csr_segment_idx, vals)
    return sums / n_segment.to(dtype=vals.dtype).clamp(min=1)


def cost_dsi_from_v_readout_dsi(
    pack, bias_gt: torch.Tensor, v_readout_dsi: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Unscaled DSI MSE (% of dsi_power); None if no complete groups.

    Peaks use baseline-subtracted absolute ``v`` (``v_readout_dsi - bias_gt``).
    """
    if pack.dsi_pos_ptr is None or int(pack.dsi_pos_ptr.numel()) <= 1:
        return None
    peak_pos_per_entry = (
        v_readout_dsi[pack.dsi_pos_entries] - bias_gt[pack.dsi_pos_entries, None]
    ).amax(dim=-1)
    peak_neg_per_entry = (
        v_readout_dsi[pack.dsi_neg_entries] - bias_gt[pack.dsi_neg_entries, None]
    ).amax(dim=-1)
    if not (torch.isfinite(peak_pos_per_entry).all() and torch.isfinite(peak_neg_per_entry).all()):
        raise RuntimeError("non-finite DSI peaks (NaN/Inf in readout)")
    peak_pos = _csr_span_mean(peak_pos_per_entry, pack.dsi_pos_ptr)
    peak_neg = _csr_span_mean(peak_neg_per_entry, pack.dsi_neg_ptr)
    v_dsi = axis_dsi_torch(peak_pos, peak_neg)
    diff = v_dsi - pack.dsi_gts
    return torch.sum(pack.dsi_scales * diff ** 2) / pack.dsi_power * 100.0


def _assemble_moving_bar_readouts(
    *,
    specs: Sequence[MovingBarSpec],
    i_sti_hex: np.ndarray,
    cost_hex_indices: Sequence[int],
    i_baseline: float,
    before_t: int,
    after_t: int,
    n_t: int,
    side: str,
    fig1: Optional[Dict[str, np.ndarray]],
    active: Sequence[str],
    nodes_for_hex_type: Callable[[int, int, str], Sequence[int]],
    waveform_mse: bool = True,
) -> Tuple[
    List[int], List[int], List[str], List[np.ndarray], List[float], List[int], List[int], int,
]:
    r_batch, r_node, r_subtype, r_readout, r_scale, r_t0, r_pd_nd = (
        [], [], [], [], [], [], [],
    )
    skipped_orthogonal = 0
    i_baseline = resolve_i_baseline(i_baseline)
    for b, spec in enumerate(specs):
        t0_by_hex: Dict[int, int] = {}
        if waveform_mse:
            for hex_idx in cost_hex_indices:
                t_first_sti = hex_first_sti_t(
                    i_sti_hex[b, :, hex_idx], i_baseline=i_baseline,
                )
                t0 = t_first_sti - before_t
                if t0 < 0 or t_first_sti + after_t > n_t:
                    raise ValueError(
                        f"cost window out of range for hex index {hex_idx} "
                        f"spec={spec.token}: t_first_sti={t_first_sti}, n_t={n_t}"
                    )
                t0_by_hex[hex_idx] = t0
        for hex_idx in cost_hex_indices:
            for subtype in active:
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
                    trace_id = fig1_trace_for_sti(side, subtype, spec)
                    if trace_id not in fig1:
                        raise KeyError(f"fig1 trace missing: {trace_id}")
                    gt_trace = fig1[trace_id]
                pd_nd_idx = PD_IDX if pref.pd_nd == "PD" else ND_IDX
                t0 = t0_by_hex.get(hex_idx, 0)
                for node_idx in nodes:
                    r_batch.append(b)
                    r_node.append(int(node_idx))
                    r_subtype.append(str(subtype))
                    if gt_trace is not None:
                        r_readout.append(gt_trace)
                    r_scale.append(1.0)
                    if waveform_mse:
                        r_t0.append(t0)
                        r_pd_nd.append(pd_nd_idx)
    return (
        r_batch, r_node, r_subtype, r_readout, r_scale, r_t0, r_pd_nd,
        skipped_orthogonal,
    )


def _pack_moving_bar_readout_tensors(
    r_batch, r_node, r_readout, r_scale, r_t0, r_pd_nd, *, device, sim_dtype,
    waveform_mse: bool = True,
):
    n = len(r_batch)
    cost_scales = torch.tensor(np.asarray(r_scale), dtype=sim_dtype, device=device)
    entry_batches = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    entry_nodes = torch.tensor(np.asarray(r_node), dtype=torch.long, device=device)
    if not waveform_mse:
        gts = torch.zeros((n, 0), dtype=sim_dtype, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return gts, cost_scales, entry_batches, entry_nodes, None, None, power
    cost_pd_nds = torch.tensor(np.asarray(r_pd_nd), dtype=torch.long, device=device)
    gts = torch.tensor(np.asarray(r_readout), dtype=sim_dtype, device=device)
    cost_t0s = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)
    power = torch.sum(cost_scales[:, None] * gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return gts, cost_scales, entry_batches, entry_nodes, cost_t0s, cost_pd_nds, power


def build_moving_bar_gt(
    connectome,
    device: Optional[str] = None,
    t_onset: int = None,
    *,
    delta_ms: float,
    fig1_path: Path = FIG1_CI_NPZ,
    use_cache: bool = True,
    bar_radius: int = BAR_RADIUS,
    multi_bar: bool = True,
    cost_radius: Optional[int] = None,
    i_baseline_moving_bar: Optional[float] = None,
    i_bright_moving_bar: Optional[float] = None,
    i_dark_moving_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
    gt_cells: Optional[Sequence[str]] = None,
    sim_dtype: torch.dtype,
    waveform_mse: bool = True,
) -> MovingBarGt:
    """Build multi-bar sti + T4/T5 cost readouts."""
    device = device or connectome.device
    side = normalize_side(connectome.meta.get("side", "right"))

    specs = gruntman_moving_bar_specs(contrasts=tuple(contrasts))
    contrast_set = frozenset(contrasts)
    i_baseline_val = resolve_i_baseline(i_baseline_moving_bar)
    peak_kw = {}
    if "bright" in contrast_set and i_bright_moving_bar is not None:
        peak_kw["i_bright_moving_bar"] = float(i_bright_moving_bar)
    if "dark" in contrast_set and i_dark_moving_bar is not None:
        peak_kw["i_dark_moving_bar"] = float(i_dark_moving_bar)
    sti = build_moving_bar_signals(
        connectome, specs=specs, t_onset=t_onset, delta_ms=delta_ms,
        bar_radius=bar_radius, multi_bar=bool(multi_bar),
        device=device, use_cache=use_cache,
        network_json=getattr(connectome, "source_json", None),
        i_baseline=i_baseline_val,
        sim_dtype=sim_dtype,
        **peak_kw,
    )
    n_t = int(sti.info["n_t"])
    fig1 = load_fig1_traces(fig1_path, delta_ms=delta_ms) if waveform_mse else None
    before_t = t_from_ms(COST_ALIGNED_FIRST_STI_MS, delta_ms=delta_ms)
    after_t = t_from_ms(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)
    n_t_cost_window = t_from_ms(COST_WINDOW_MS, delta_ms=delta_ms) + 1

    active = active_gt_cells(
        gt_cells, GT_CELLS, connectome.cells, context="moving_bar",
    )

    node_cell = node_cells(connectome)
    idx_from_uv = {
        (int(hex.u), int(hex.v)): hex_idx
        for hex_idx, hex in enumerate(sti_hexes(connectome))
    }
    hexes = moving_bar_cost_hexes(connectome, cost_radius=cost_radius)
    center_hex = hexes[0] if cost_radius == 0 and len(hexes) == 1 else None
    cost_hex_indices = [idx_from_uv[(int(hex.u), int(hex.v))] for hex in hexes]
    hex_by_idx = {hex_idx: hex for hex, hex_idx in zip(hexes, cost_hex_indices)}

    def _nodes_for_hex_type(_b, hex_idx, subtype):
        hex = hex_by_idx[hex_idx]
        return hex2gt(connectome, hex.u, hex.v, subtype, node_cell)

    rows = _assemble_moving_bar_readouts(
        specs=sti.specs,
        i_sti_hex=sti.i_sti_hex,
        cost_hex_indices=cost_hex_indices,
        i_baseline=i_baseline_val,
        before_t=before_t,
        after_t=after_t,
        n_t=n_t,
        side=side,
        fig1=fig1,
        active=active,
        nodes_for_hex_type=_nodes_for_hex_type,
        waveform_mse=waveform_mse,
    )
    (
        r_batch, r_node, r_subtype, r_readout, r_scale, r_t0, r_pd_nd,
        skipped_orthogonal,
    ) = rows

    if not r_batch:
        raise ValueError("no moving-bar cost nodes (check subtypes and sti hexes)")

    gts, cost_scales, entry_batches, entry_nodes, cost_t0s, cost_pd_nds, power = (
        _pack_moving_bar_readout_tensors(
            r_batch, r_node, r_readout, r_scale, r_t0, r_pd_nd,
            device=device, sim_dtype=sim_dtype, waveform_mse=waveform_mse,
        )
    )
    (
        dsi_pos_entries, dsi_neg_entries, dsi_pos_ptr, dsi_neg_ptr,
        dsi_tgt, dsi_w, dsi_pow,
    ) = pack_moving_bar_dsi_tensors(
        *assemble_moving_bar_dsi_groups(
            sti.specs, r_batch, r_subtype, r_scale, side=side,
        ),
        device=device,
        sim_dtype=sim_dtype,
    )

    info = {
        **sti.info,
        "n_cost": int(entry_batches.shape[0]),
        "n_cost_pd": int((cost_pd_nds == PD_IDX).sum().item()) if cost_pd_nds is not None else 0,
        "n_cost_nd": int((cost_pd_nds == ND_IDX).sum().item()) if cost_pd_nds is not None else 0,
        "n_cost_dsi": int(dsi_tgt.shape[0]),
        "n_batch": sti.info["n_batch"],
        "n_cost_hexes": len(hexes),
        "cost_radius": cost_radius,
        "cost_hex_uv": (int(center_hex.u), int(center_hex.v)) if center_hex else None,
        "side": side,
        "active_gts": active,
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
            "cost_window_t": n_t_cost_window,
        })
    return MovingBarGt(
        i_sti=sti.i_sti,
        gts=gts,
        power=power,
        cost_scales=cost_scales,
        cost_t0s=cost_t0s,
        entry_batches=entry_batches,
        entry_nodes=entry_nodes,
        cost_pd_nds=cost_pd_nds,
        n_batch=sti.info["n_batch"],
        n_t=n_t,
        info=info,
        dsi_pos_entries=dsi_pos_entries,
        dsi_neg_entries=dsi_neg_entries,
        dsi_pos_ptr=dsi_pos_ptr,
        dsi_neg_ptr=dsi_neg_ptr,
        dsi_gts=dsi_tgt,
        dsi_scales=dsi_w,
        dsi_power=dsi_pow,
    )


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
    cost_radius,
    n_t: int,
    *,
    at_x=None,
    at_y=None,
    t_onset: int = None,
    delta_ms: float,
) -> MovingBarSessionT0:
    """Session-level ``t0`` / horizon grids for moving-bar cost or analyze."""
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("moving_bar_session_t0_grids requires session.backend.network")
    i_baseline = resolve_moving_bar_i_baseline(session.train_opts)

    side = normalize_side(connectome.meta.get('side', 'right'))
    hexes = moving_bar_cost_hexes(connectome, cost_radius=cost_radius)
    if at_x is not None or at_y is not None:
        filt_hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
        if not filt_hexes:
            raise SystemExit(
                f'no sti hexes match x={at_x!r} y={at_y!r} within cost_radius',
            )
    else:
        filt_hexes = hexes
    sti = build_moving_bar_signals(
        connectome, specs=specs, n_t=n_t, t_onset=t_onset, delta_ms=delta_ms,
        device=connectome.node_cells.device, i_baseline=i_baseline,
    )
    idx_from_uv = {
        (int(hex.u), int(hex.v)): hex_idx
        for hex_idx, hex in enumerate(sti_hexes(connectome))
    }
    hex_indices = [idx_from_uv[(int(hex.u), int(hex.v))] for hex in hexes]
    filt_hex_indices = [idx_from_uv[(int(hex.u), int(hex.v))] for hex in filt_hexes]
    grids = build_moving_bar_t0_grids(
        sti.i_sti_hex, specs, n_t, i_baseline,
        hex_indices=hex_indices,
        filt_hex_indices=filt_hex_indices,
        connectome=connectome,
        filt_network_hexes=filt_hexes,
    )
    return MovingBarSessionT0(
        t0_bn=grids.t0_bn,
        before_t=grids.before_t,
        after_t=grids.after_t,
        side=side,
        n_filter_hexes=len(filt_hexes),
    )


def _pack_cells(session, task: str) -> List[str]:
    """Unique cells on ``pack.entry_nodes`` (pack order)."""
    pack = session.pack_for(task)
    entry_nodes = pack.entry_nodes
    if torch.is_tensor(entry_nodes):
        entry_nodes = entry_nodes.detach().cpu().numpy()
    entry_nodes = np.asarray(entry_nodes, dtype=np.int64)
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("_pack_cells requires session.backend.network")
    node_cells = connectome.node_cells[entry_nodes]
    if torch.is_tensor(node_cells):
        node_cells = node_cells.detach().cpu().numpy()
    cells = list(connectome.cells)
    seq = [str(cells[int(ti)]) for ti in node_cells]
    seen: set = set()
    out: List[str] = []
    for name in seq:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def moving_bar_specs_by_cell(session, task: str, side: str) -> Dict[str, List[str]]:
    """Per-readout-cell active bar spec tokens for ``side`` and task contrast."""
    contrast = "bright" if "bright" in task else "dark"
    return {
        st: [
            f'{d}_{c}_{w}'
            for d, c, w in active_stis_for_subtype(side, st)
            if c == contrast
        ]
        for st in _pack_cells(session, task)
    }


def build_moving_bar_sti_opts(
    contrast: str,
    *,
    i_baseline_moving_bar,
    i_moving_bar,
    ms_pre,
    delta_ms,
    delta_ms_pre,
    multi_bar: bool,
    gt_cells=None,
):
    """Sti moving-bar sti opts for ``moving_bar_{contrast}``."""
    if contrast not in MOVING_BAR_CONTRASTS:
        raise ValueError(f"moving-bar contrast must be 'bright' or 'dark', got {contrast!r}")
    i_peak = "i_bright_moving_bar" if contrast == "bright" else "i_dark_moving_bar"
    out = {
        "i_baseline_moving_bar": i_baseline_moving_bar,
        i_peak: i_moving_bar,
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "multi_bar": bool(multi_bar),
    }
    rs = normalize_gt_cells(gt_cells)
    if rs is not None:
        out["gt_cells"] = rs
    return out


def session_moving_bar_i_baseline(train_opts) -> float:
    """``i_baseline_moving_bar`` from moving-bar sti opts on a train session."""
    return resolve_moving_bar_i_baseline(train_opts)


def enrich_moving_bar_sti_opts(opts, info, *, cost_radius):
    """Attach runtime fields from a built moving-bar task; keep canonical ``i_*``."""
    out = dict(opts)
    out["n_t"] = int(info["n_t"])
    if out.get("delta_ms") is None:
        raise ValueError("moving-bar sti opts require delta_ms")
    if out.get("delta_ms_pre") is None:
        raise ValueError("moving-bar sti opts require delta_ms_pre")
    dt = float(out["delta_ms"])
    dt_pre = float(out["delta_ms_pre"])
    out["ms_pre"] = float(info["t_onset"]) * dt_pre
    out["spec_tokens"] = list(info["spec_tokens"])
    if cost_radius is not None:
        out["cost_radius"] = int(cost_radius)
    out["delta_ms_pre"] = dt_pre
    if "active_gts" in info:
        out["gt_cells"] = list(info["active_gts"])
    return out
