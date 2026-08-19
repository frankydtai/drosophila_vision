# -*- coding: utf-8 -*-
"""Moving-bar pack: bind GT numbers to the network for cost.

Cost hexes, sti ``i_sti``, :class:`MbarGt` packing, and DSI entry CSR.
GT traces and motion preference come from :mod:`task.mbar.gt`.
:class:`~task.spread.pack.Pack` assembly lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from neuron.borst import t_from_ms
from network.construction import (
    active_gt_cells,
    node_cells,
    gt_cells_from_opts,
    standardize_cost_radius,
)
from task.spread.pack import Pack, CostPartPlotSpec, cost_hex_label
from task.spread.sti_spec import i_baseline_from_i_sti
from task.spread.gt import contrast_sign
from task.mbar.gt import (
    FIG1_CI_NPZ,
    GT_CELLS,
    active_stis_from_subtype,
    axis_dsi_torch,
    fig1_trace_from_sti,
    hardcoded_axis_dsi,
    load_fig1_traces,
    motion_preference,
    w_token,
)
from task.mbar.sti_geo import (
    BAR_RADIUS,
    _as_int64_np,
    filter_sti_hexes,
    mbar_cost_hexes,
    network_uv_np,
    sti_hexes,
)
from task.mbar.sti_spec import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_AFTER_MS,
    MbarSpec,
    ND_IDX,
    PD_IDX,
    build_mbar_signals,
    build_mbar_t0_grids,
    gruntman_mbar_specs,
    hex_first_sti_t,
)



@dataclass
class MbarGt:
    i_sti: torch.Tensor
    gts: torch.Tensor
    power: torch.Tensor
    cost_scales: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    n_b: int
    n_t: int
    n_cost_hex: int
    active_gts: List[str]
    waveform_mse: bool
    spec_tokens: List[str]
    cost_t0s: Optional[torch.Tensor] = None
    cost_pd_nds: Optional[torch.Tensor] = None
    dsi_pos_entries: Optional[torch.Tensor] = None
    dsi_neg_entries: Optional[torch.Tensor] = None
    dsi_pos_ptr: Optional[torch.Tensor] = None
    dsi_neg_ptr: Optional[torch.Tensor] = None
    dsi_gts: Optional[torch.Tensor] = None
    dsi_scales: Optional[torch.Tensor] = None
    dsi_power: Optional[torch.Tensor] = None
    entry_part_keys: Tuple[str, ...] = ()


def cell_part_key(contrast: str, cell: str, pd_nd_label: str) -> str:
    return f"mbar_{contrast}_{cell}_{pd_nd_label}"


def dsi_part_key(contrast: str) -> str:
    return f"mbar_{contrast}_DSI"


def _mbar_cost_part_plot_specs(
    entry_part_keys: Sequence[str],
    entry_nodes: torch.Tensor,
    cost_scales: torch.Tensor,
    cost_pd_nds: torch.Tensor,
    connectome,
    contrast: str,
    dsi_part_key_val: Optional[str],
) -> Dict[str, CostPartPlotSpec]:
    specs: Dict[str, CostPartPlotSpec] = {}
    pd_nd = cost_pd_nds.detach().cpu().numpy()
    node_cells = connectome.node_cells[entry_nodes].detach().cpu().numpy()
    cells = connectome.cells
    for entry, part_key in enumerate(entry_part_keys):
        if float(cost_scales[entry]) <= 0.0:
            continue
        if part_key in specs:
            continue
        pd_nd_label = PD_ND_LABELS[int(pd_nd[entry])]
        cell = str(cells[int(node_cells[entry])])
        label = (
            f"{pd_nd_label} ({contrast})" if contrast else pd_nd_label
        )
        specs[part_key] = CostPartPlotSpec(
            part_key, cell, ("mbar_pd_nd", pd_nd_label), label,
        )
    if dsi_part_key_val and dsi_part_key_val not in specs:
        specs[dsi_part_key_val] = CostPartPlotSpec(
            dsi_part_key_val,
            None,
            ("mbar_pd_nd", "DSI"),
            f"DSI ({contrast})" if contrast else "DSI",
        )
    return specs


def nodes_from_hexes(connectome, cell: str, hexes: Sequence) -> np.ndarray:
    """Nodes of ``cell`` whose hex is among ``hexes`` (vectorized axial uv pack)."""
    if not hexes:
        return np.zeros(0, dtype=np.int64)
    if cell not in connectome.cells:
        raise ValueError(f"unknown cell {cell!r}; known: {list(connectome.cells)}")
    cell_idx = int(connectome.cells.index(cell))
    node_u_np, node_v_np = network_uv_np(connectome)
    cell_idxs = _as_int64_np(connectome.node_cells)
    uv_span = int(max(np.max(np.abs(node_u_np)), np.max(np.abs(node_v_np)), 1)) + 1
    pack = (node_u_np + uv_span) * (2 * uv_span + 1) + (node_v_np + uv_span)
    hex_pack = np.array(
        [
            (int(hex.u) + uv_span) * (2 * uv_span + 1) + (int(hex.v) + uv_span)
            for hex in hexes
        ],
        dtype=np.int64,
    )
    return np.where((cell_idxs == cell_idx) & np.isin(pack, hex_pack))[0].astype(np.int64)


def filter_requested_specs(
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> List[str]:
    """Keep ``requested`` specs that exist in ``available``; omit ``requested`` -> all."""
    avail = list(available)
    if requested is None:
        return avail
    missing = [token for token in requested if token not in avail]
    if missing:
        raise ValueError(f"spec(s) {missing} not in {avail}")
    return list(requested)


def assemble_mbar_dsi_groups(
    specs: Sequence[MbarSpec],
    entry_bs: Sequence[int],
    entry_cells: Sequence[str],
    entry_cost_scales: Sequence[float],
    *,
    side: str,
) -> Tuple[List[List[int]], List[List[int]], List[float], List[float]]:
    """One DSI group per ``(cell, contrast, w_token, axis)``."""
    bs_by_condition: dict[tuple[str, str, str], list[int]] = {}
    for b, spec in enumerate(specs):
        key = (spec.direction, spec.contrast, w_token(spec.w_deg))
        bs_by_condition.setdefault(key, []).append(b)

    entries_by_cell_b: dict[tuple[str, int], list[int]] = {}
    for entry, (b, cell) in enumerate(zip(entry_bs, entry_cells)):
        entries_by_cell_b.setdefault((str(cell), int(b)), []).append(entry)

    pos_groups: List[List[int]] = []
    neg_groups: List[List[int]] = []
    dsi_vals: List[float] = []
    scales: List[float] = []
    cells = sorted({str(cell) for cell in entry_cells})
    for cell in cells:
        for pos_dir, neg_dir in (("right", "left"), ("up", "down")):
            contrast_ws = {
                (contrast, w_token)
                for (direction, contrast, w_token) in bs_by_condition
                if direction in (pos_dir, neg_dir)
            }
            for contrast, w_token in sorted(contrast_ws):
                pos_bs = bs_by_condition.get((pos_dir, contrast, w_token), [])
                neg_bs = bs_by_condition.get((neg_dir, contrast, w_token), [])
                if not pos_bs or not neg_bs:
                    continue
                pos_entries: list[int] = []
                for pb in pos_bs:
                    pos_entries.extend(entries_by_cell_b.get((cell, pb), []))
                neg_entries: list[int] = []
                for nb in neg_bs:
                    neg_entries.extend(entries_by_cell_b.get((cell, nb), []))
                if not pos_entries or not neg_entries:
                    continue
                dsi = hardcoded_axis_dsi(side, cell, specs[pos_bs[0]])
                if dsi is None:
                    continue
                w_pos = float(np.mean([float(entry_cost_scales[entry]) for entry in pos_entries]))
                w_neg = float(np.mean([float(entry_cost_scales[entry]) for entry in neg_entries]))
                pos_groups.append(pos_entries)
                neg_groups.append(neg_entries)
                dsi_vals.append(float(dsi))
                scales.append(0.5 * (w_pos + w_neg))
    return pos_groups, neg_groups, dsi_vals, scales


def _csr_from_groups(
    groups: Sequence[Sequence[int]], *, device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(flat_entries, ptr)`` with ``ptr`` length ``n_group + 1``."""
    ptr = [0]
    flat: list[int] = []
    for dsi_group in groups:
        flat.extend(int(entry) for entry in dsi_group)
        ptr.append(len(flat))
    entries_t = torch.tensor(np.asarray(flat, dtype=np.int64), dtype=torch.long, device=device)
    ptr_t = torch.tensor(np.asarray(ptr, dtype=np.int64), dtype=torch.long, device=device)
    return entries_t, ptr_t


def pack_mbar_dsi(pos_groups, neg_groups, dsi_vals, scales, *, device, sim_dtype):
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
    """Remap CSR DSI groups onto kept cost entries; drop incomplete groups."""
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
    n = int(pack.entry_bs.shape[0])
    kept = torch.as_tensor(kept_old_entries, dtype=torch.long, device=device)
    lut = torch.full((n,), -1, dtype=torch.long, device=device)
    lut[kept] = torch.arange(kept.numel(), dtype=torch.long, device=device)

    n_dsi = int(pack.dsi_pos_ptr.numel()) - 1
    new_pos_groups: list[list[int]] = []
    new_neg_groups: list[list[int]] = []
    kept_dsi_group_idxs: list[int] = []
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
        kept_dsi_group_idxs.append(dsi_group_idx)
    if not kept_dsi_group_idxs:
        return _empty_dsi_fields(pack, device)
    dsi_pos_entries, dsi_pos_ptr = _csr_from_groups(new_pos_groups, device=device)
    dsi_neg_entries, dsi_neg_ptr = _csr_from_groups(new_neg_groups, device=device)
    kept_dsi_group_idx = torch.tensor(kept_dsi_group_idxs, dtype=torch.long, device=device)
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
    """Mean of ``vals`` over CSR spans defined by ``ptr``."""
    n_g = int(ptr.numel()) - 1
    if n_g == 0:
        return vals.new_zeros((0,))
    n_ptr = ptr[1:] - ptr[:-1]
    csr_idx = torch.repeat_interleave(
        torch.arange(n_g, device=vals.device, dtype=torch.long),
        n_ptr,
    )
    sums = vals.new_zeros((n_g,))
    sums.scatter_add_(0, csr_idx, vals)
    return sums / n_ptr.to(dtype=vals.dtype).clamp(min=1)


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


def _assemble_mbar_readouts(
    *,
    specs: Sequence[MbarSpec],
    i_sti_hex: np.ndarray,
    cost_hex_idxs: Sequence[int],
    i_baseline: float,
    before_t: int,
    after_t: int,
    n_t: int,
    side: str,
    fig1: Optional[Dict[str, np.ndarray]],
    active: Sequence[str],
    nodes_from_hex_type: Callable[[int, int, str], Sequence[int]],
    waveform_mse: bool = True,
) -> Tuple[
    List[int], List[int], List[str], List[np.ndarray], List[float], List[int], List[int], List[str], int,
]:
    entry_bs, entry_nodes, entry_cells, entry_gts, entry_cost_scales, entry_cost_t0s, entry_cost_pd_nds, entry_part_keys = (
        [], [], [], [], [], [], [], [],
    )
    skipped_orthogonal = 0
    i_baseline = float(i_baseline)
    for b, spec in enumerate(specs):
        t0_by_hex: Dict[int, int] = {}
        if waveform_mse:
            for hex_idx in cost_hex_idxs:
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
        for hex_idx in cost_hex_idxs:
            for cell in active:
                pref = motion_preference(side, cell, spec.direction, spec.contrast)
                if pref is None:
                    skipped_orthogonal += 1
                    continue
                nodes = nodes_from_hex_type(b, hex_idx, cell)
                if len(nodes) == 0:
                    continue
                gt_trace = None
                if waveform_mse:
                    if fig1 is None:
                        raise ValueError("fig1 traces required when waveform_mse=True")
                    trace_token = fig1_trace_from_sti(side, cell, spec)
                    if trace_token not in fig1:
                        raise KeyError(f"fig1 trace missing: {trace_token}")
                    gt_trace = fig1[trace_token]
                pd_nd_idx = PD_IDX if pref.pd_nd == "PD" else ND_IDX
                pd_nd_label = PD_ND_LABELS[pd_nd_idx]
                t0 = t0_by_hex.get(hex_idx, 0)
                for node in nodes:
                    entry_bs.append(b)
                    entry_nodes.append(int(node))
                    entry_cells.append(str(cell))
                    if gt_trace is not None:
                        entry_gts.append(gt_trace)
                    entry_cost_scales.append(1.0)
                    if waveform_mse:
                        entry_cost_t0s.append(t0)
                        entry_cost_pd_nds.append(pd_nd_idx)
                        entry_part_keys.append(cell_part_key(spec.contrast, cell, pd_nd_label))
    return (
        entry_bs, entry_nodes, entry_cells, entry_gts, entry_cost_scales,
        entry_cost_t0s, entry_cost_pd_nds, entry_part_keys,
        skipped_orthogonal,
    )


def _pack_mbar_entries(
    entry_bs, entry_nodes, entry_gts, entry_cost_scales, entry_cost_t0s, entry_cost_pd_nds,
    *, device, sim_dtype,
    waveform_mse: bool = True,
):
    n = len(entry_bs)
    cost_scales = torch.tensor(np.asarray(entry_cost_scales), dtype=sim_dtype, device=device)
    entry_bs = torch.tensor(np.asarray(entry_bs), dtype=torch.long, device=device)
    entry_nodes = torch.tensor(np.asarray(entry_nodes), dtype=torch.long, device=device)
    if not waveform_mse:
        gts = torch.zeros((n, 0), dtype=sim_dtype, device=device)
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
        return gts, cost_scales, entry_bs, entry_nodes, None, None, power
    cost_pd_nds = torch.tensor(np.asarray(entry_cost_pd_nds), dtype=torch.long, device=device)
    gts = torch.tensor(np.asarray(entry_gts), dtype=sim_dtype, device=device)
    cost_t0s = torch.tensor(np.asarray(entry_cost_t0s), dtype=torch.long, device=device)
    power = torch.sum(cost_scales[:, None] * gts ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return gts, cost_scales, entry_bs, entry_nodes, cost_t0s, cost_pd_nds, power


def build_mbar_gt(
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
    i_baseline: float,
    i_sti: float,
    contrasts: Sequence[str],
    gt_cells: Optional[Sequence[str]] = None,
    sim_dtype: torch.dtype,
    waveform_mse: bool = True,
) -> MbarGt:
    """Build multi-bar sti + T4/T5 cost readouts."""
    device = device or connectome.device
    side = connectome.meta.get("side", "right")

    specs = gruntman_mbar_specs(contrasts=tuple(contrasts))
    i_baseline_val = float(i_baseline)
    sti = build_mbar_signals(
        connectome, specs=specs, t_onset=t_onset, delta_ms=delta_ms,
        bar_radius=bar_radius, multi_bar=bool(multi_bar),
        device=device, use_cache=use_cache,
        network_json=getattr(connectome, "source_json", None),
        i_baseline=i_baseline_val,
        i_sti=float(i_sti),
        sim_dtype=sim_dtype,
    )
    n_t = int(sti.n_t)
    fig1 = load_fig1_traces(fig1_path, delta_ms=delta_ms) if waveform_mse else None
    before_t = t_from_ms(COST_ALIGNED_FIRST_STI_MS, delta_ms=delta_ms)
    after_t = t_from_ms(COST_WINDOW_AFTER_MS, delta_ms=delta_ms)

    active = active_gt_cells(
        gt_cells, GT_CELLS, connectome.cells, context="mbar",
    )

    cells = node_cells(connectome)
    hex_idx = {
        (int(hex.u), int(hex.v)): hex_idx
        for hex_idx, hex in enumerate(sti_hexes(connectome))
    }
    hexes = mbar_cost_hexes(connectome, cost_radius=cost_radius)
    cost_hex_idxs = [hex_idx[(int(hex.u), int(hex.v))] for hex in hexes]
    hex_by_idx = {hex_idx: hex for hex, hex_idx in zip(hexes, cost_hex_idxs)}

    def _nodes_from_hex_type(b, hex_idx, cell):
        hex = hex_by_idx[hex_idx]
        return connectome.nodes_at_uv(hex.u, hex.v, cell, cells=cells)

    rows = _assemble_mbar_readouts(
        specs=sti.specs,
        i_sti_hex=sti.i_sti_hex,
        cost_hex_idxs=cost_hex_idxs,
        i_baseline=i_baseline_val,
        before_t=before_t,
        after_t=after_t,
        n_t=n_t,
        side=side,
        fig1=fig1,
        active=active,
        nodes_from_hex_type=_nodes_from_hex_type,
        waveform_mse=waveform_mse,
    )
    (
        entry_bs, entry_nodes, entry_cells, entry_gts, entry_cost_scales,
        entry_cost_t0s, entry_cost_pd_nds, entry_part_keys,
        _,
    ) = rows

    if not entry_bs:
        raise ValueError("no moving-bar cost nodes (check subtypes and sti hexes)")

    gts, cost_scales, entry_bs, entry_nodes, cost_t0s, cost_pd_nds, power = (
        _pack_mbar_entries(
            entry_bs, entry_nodes, entry_gts, entry_cost_scales,
            entry_cost_t0s, entry_cost_pd_nds,
            device=device, sim_dtype=sim_dtype, waveform_mse=waveform_mse,
        )
    )
    (
        dsi_pos_entries, dsi_neg_entries, dsi_pos_ptr, dsi_neg_ptr,
        dsi_tgt, dsi_w, dsi_pow,
    ) = pack_mbar_dsi(
        *assemble_mbar_dsi_groups(
            sti.specs, entry_bs.tolist(), entry_cells, cost_scales.tolist(), side=side,
        ),
        device=device,
        sim_dtype=sim_dtype,
    )

    return MbarGt(
        i_sti=sti.i_sti,
        gts=gts,
        power=power,
        cost_scales=cost_scales,
        cost_t0s=cost_t0s,
        entry_bs=entry_bs,
        entry_nodes=entry_nodes,
        cost_pd_nds=cost_pd_nds,
        n_b=sti.n_b,
        n_t=n_t,
        n_cost_hex=len(hexes),
        active_gts=list(active),
        waveform_mse=bool(waveform_mse),
        spec_tokens=[spec.token for spec in sti.specs],
        dsi_pos_entries=dsi_pos_entries,
        dsi_neg_entries=dsi_neg_entries,
        dsi_pos_ptr=dsi_pos_ptr,
        dsi_neg_ptr=dsi_neg_ptr,
        dsi_gts=dsi_tgt,
        dsi_scales=dsi_w,
        dsi_power=dsi_pow,
        entry_part_keys=tuple(entry_part_keys),
    )


@dataclass
class MbarSessionT0:
    t0_bn: np.ndarray
    before_t: Dict[str, int]
    after_t: Dict[str, int]
    side: str
    n_hex: int


def bar_specs_from_task(session, task, contrast) -> List[MbarSpec]:
    """Gruntman bar specs for ``task``×``contrast``."""
    contrast_sign(contrast)
    return list(gruntman_mbar_specs(contrasts=(contrast,)))


def mbar_session_t0_grids(
    session,
    specs: Sequence[MbarSpec],
    cost_radius,
    n_t: int,
    *,
    at_x=None,
    at_y=None,
    t_onset: int = None,
    delta_ms: float,
) -> MbarSessionT0:
    """Session-level ``t0`` / horizon grids for moving-bar cost or analyze."""
    connectome = session.connectome
    i_sti = (session.train_opts or {}).get("i_sti") or {}
    i_baseline = i_baseline_from_i_sti(i_sti)

    side = connectome.meta.get('side', 'right')
    hexes = mbar_cost_hexes(connectome, cost_radius=cost_radius)
    if at_x is not None or at_y is not None:
        filt_hexes = filter_sti_hexes(hexes, at_x=at_x, at_y=at_y)
        if not filt_hexes:
            raise SystemExit(
                f'no sti hexes match x={at_x!r} y={at_y!r} within cost_radius',
            )
    else:
        filt_hexes = hexes
    contrast = specs[0].contrast if specs else "bright"
    if contrast not in i_sti:
        raise ValueError(
            f"train opts i_sti missing contrast {contrast!r}"
        )
    sti = build_mbar_signals(
        connectome, specs=specs, n_t=n_t, t_onset=t_onset, delta_ms=delta_ms,
        device=connectome.node_cells.device, i_baseline=i_baseline,
        i_sti=float(i_sti[contrast]),
        sim_dtype=session.sim_dtype,
    )
    hex_idx = {
        (int(hex.u), int(hex.v)): hex_idx
        for hex_idx, hex in enumerate(sti_hexes(connectome))
    }
    hex_idxs = [hex_idx[(int(hex.u), int(hex.v))] for hex in hexes]
    filt_hex_idxs = [hex_idx[(int(hex.u), int(hex.v))] for hex in filt_hexes]
    grids = build_mbar_t0_grids(
        sti.i_sti_hex, specs, n_t, i_baseline,
        hex_idxs=hex_idxs,
        filt_hex_idxs=filt_hex_idxs,
        connectome=connectome,
        filt_network_hexes=filt_hexes,
    )
    return MbarSessionT0(
        t0_bn=grids.t0_bn,
        before_t=grids.before_t,
        after_t=grids.after_t,
        side=side,
        n_hex=len(filt_hexes),
    )


def _pack_cells(session, task: str, contrast: str) -> List[str]:
    """Unique cells on ``pack.entry_nodes`` (pack order)."""
    pack = session.packs[task][contrast]
    entry_nodes = pack.entry_nodes
    if torch.is_tensor(entry_nodes):
        entry_nodes = entry_nodes.detach().cpu().numpy()
    entry_nodes = np.asarray(entry_nodes, dtype=np.int64)
    connectome = session.connectome
    node_cells = connectome.node_cells[entry_nodes]
    if torch.is_tensor(node_cells):
        node_cells = node_cells.detach().cpu().numpy()
    return list(dict.fromkeys(
        str(connectome.cells[int(cell_idx)]) for cell_idx in node_cells
    ))


def mbar_specs_by_cell(session, task: str, contrast: str, side: str) -> Dict[str, List[str]]:
    """Per-readout-cell active bar spec tokens for ``side`` and task contrast."""
    contrast_sign(contrast)
    return {
        cell: [
            f'{direction}_{active_contrast}_{w_token}'
            for direction, active_contrast, w_token in active_stis_from_subtype(side, cell)
            if active_contrast == contrast
        ]
        for cell in _pack_cells(session, task, contrast)
    }


def build_mbar_sti_opts(
    *,
    ms_pre,
    delta_ms,
    delta_ms_pre,
    multi_bar: bool,
    gt_cells=None,
):
    """Moving-bar sti opts: timing / geometry only (currents live on session ``i_sti``)."""
    sti_opts = {
        "ms_pre": ms_pre,
        "delta_ms": delta_ms,
        "delta_ms_pre": delta_ms_pre,
        "multi_bar": bool(multi_bar),
    }
    if gt_cells is not None:
        sti_opts["gt_cells"] = list(gt_cells)
    return sti_opts


def resolve_mbar_sti_opts(opts, **_):
    return build_mbar_sti_opts(
        ms_pre=opts["ms_pre"],
        delta_ms=opts["delta_ms"],
        delta_ms_pre=opts["delta_ms_pre"],
        multi_bar=opts["multi_bar"],
        gt_cells=opts.get("gt_cells"),
    )


def mbar_sti_opts(
    mbar_sti_opts: Optional[dict],
    *,
    ms_pre,
    delta_ms: float,
    delta_ms_pre: float,
    multi_bar: bool,
) -> dict:
    """Build moving-bar sti opts when CLI sidecar is absent."""
    if mbar_sti_opts:
        return dict(mbar_sti_opts)
    return build_mbar_sti_opts(
        ms_pre=ms_pre,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        multi_bar=multi_bar,
    )


def build_mbar_pack(
    connectome,
    *,
    contrast: str,
    gt_amp: float,
    device: str,
    sim_dtype: torch.dtype,
    i_sti: Dict[str, float],
    mbar_sti_opts: Optional[dict],
    filter: str,
    delta_ms: float,
    delta_ms_pre: float,
    ms_pre,
    multi_bar: bool,
):
    task = "mbar"
    device = device or connectome.device
    opts = mbar_sti_opts(
        mbar_sti_opts,
        ms_pre=ms_pre,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        multi_bar=multi_bar,
    )
    cost_radius = standardize_cost_radius(opts.get("cost_radius"))
    mbar_gt = build_mbar_gt(
        connectome=connectome,
        device=device,
        sim_dtype=sim_dtype,
        t_onset=t_from_ms(
            float(opts["ms_pre"]),
            delta_ms=float(opts["delta_ms_pre"]),
        ),
        delta_ms=float(opts["delta_ms"]),
        cost_radius=cost_radius,
        i_baseline=i_baseline_from_i_sti(i_sti),
        i_sti=float(i_sti[contrast]),
        contrasts=(contrast,),
        gt_cells=gt_cells_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", multi_bar)),
        waveform_mse=True,
    )
    sti_opts = dict(opts)
    sti_opts["n_t"] = int(mbar_gt.n_t)
    sti_opts["spec_tokens"] = list(mbar_gt.spec_tokens)
    if cost_radius is not None:
        sti_opts["cost_radius"] = int(cost_radius)
    sti_opts["gt_cells"] = list(mbar_gt.active_gts)
    dsi_key = (
        dsi_part_key(contrast)
        if mbar_gt.dsi_pos_ptr is not None
        and int(mbar_gt.dsi_pos_ptr.numel()) > 1
        else None
    )
    pack = Pack(
        task=task,
        contrast=contrast,
        i_sti=mbar_gt.i_sti,
        gts=mbar_gt.gts,
        power=mbar_gt.power,
        cost_scales=mbar_gt.cost_scales,
        entry_bs=mbar_gt.entry_bs,
        entry_nodes=mbar_gt.entry_nodes,
        cost_t0s=mbar_gt.cost_t0s,
        cost_radius=cost_radius,
        cost_pd_nds=mbar_gt.cost_pd_nds,
        dsi_pos_entries=mbar_gt.dsi_pos_entries,
        dsi_neg_entries=mbar_gt.dsi_neg_entries,
        dsi_pos_ptr=mbar_gt.dsi_pos_ptr,
        dsi_neg_ptr=mbar_gt.dsi_neg_ptr,
        dsi_gts=mbar_gt.dsi_gts,
        dsi_scales=mbar_gt.dsi_scales,
        dsi_power=mbar_gt.dsi_power,
        waveform_mse=bool(mbar_gt.waveform_mse),
        entry_part_keys=mbar_gt.entry_part_keys,
        dsi_part_key=dsi_key,
        cost_part_plot_specs=(
            _mbar_cost_part_plot_specs(
                mbar_gt.entry_part_keys,
                mbar_gt.entry_nodes,
                mbar_gt.cost_scales,
                mbar_gt.cost_pd_nds,
                connectome,
                contrast,
                dsi_key,
            )
            if mbar_gt.cost_pd_nds is not None
            else None
        ),
    )
    hex_label = cost_hex_label(cost_radius, mbar_gt.n_cost_hex)
    label = (
        f"moving-bar {contrast} (B={mbar_gt.n_b} stis, "
        f"{int(mbar_gt.entry_bs.shape[0])} cost nodes, {hex_label})"
    )
    return pack, sti_opts, label
