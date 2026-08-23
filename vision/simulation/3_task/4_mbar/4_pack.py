# -*- coding: utf-8 -*-
"""Moving-bar pack: bind GT numbers to the network for cost.

Cost hexes, sti ``i_sti``, and :class:`MbarGt` packing.
GT traces and motion preference come from :mod:`task.mbar.gt`.
:class:`MbarPack` is mbar-specific.
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
from task.spread.pack import CostPartPlotSpec, cost_hex_label, cost_sti_hexes
from task.spread.sti_spec import i_baseline_from_i_sti
from task.spread.gt import contrast_sign
from task.mbar.gt import (
    FIG1_CI_NPZ,
    GT_CELLS,
    active_stis_from_subtype,
    fig1_trace_from_sti,
    load_fig1_traces,
    motion_preference,
    w_token,
)
from task.mbar.sti_geo import (
    BAR_RADIUS,
    sti_hexes_at_xy,
    node_us_vs,
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
    entry_part_keys: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MbarPack:
    """One mbar train pack: task×contrast drive + entries + gts."""

    task: str
    contrast: str
    i_sti: torch.Tensor
    gts: torch.Tensor
    cost_scales: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor
    cost_t0s: Optional[torch.Tensor] = None
    cost_radius: Optional[int] = None
    cost_pd_nds: Optional[torch.Tensor] = None
    entry_part_keys: Tuple[str, ...] = ()
    cost_part_plot_specs: Optional[Dict[str, CostPartPlotSpec]] = None


def cell_part_key(contrast: str, cell: str, pd_nd_label: str) -> str:
    return f"mbar_{contrast}_{cell}_{pd_nd_label}"


def _mbar_cost_part_plot_specs(
    entry_part_keys: Sequence[str],
    entry_nodes: torch.Tensor,
    cost_scales: torch.Tensor,
    cost_pd_nds: torch.Tensor,
    connectome,
    contrast: str,
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
        specs[part_key] = CostPartPlotSpec(
            part_key,
            cell,
            ("mbar_pd_nd", pd_nd_label),
            f"{pd_nd_label} ({contrast})" if contrast else pd_nd_label,
        )
    return specs


def nodes_from_hexes(connectome, cell: str, hexes: Sequence) -> np.ndarray:
    """Nodes of ``cell`` whose hex is among ``hexes`` (vectorized axial uv pack)."""
    if not hexes:
        return np.zeros(0, dtype=np.int64)
    if cell not in connectome.cells:
        raise ValueError(f"unknown cell {cell!r}; known: {list(connectome.cells)}")
    cell_idx = int(connectome.cells.index(cell))
    node_us, node_vs = node_us_vs(connectome)
    cell_idxs = np.array(connectome.node_cells, dtype=np.int64)
    uv_span = int(max(np.max(np.abs(node_us)), np.max(np.abs(node_vs)), 1)) + 1
    return np.where(
        (cell_idxs == cell_idx)
        & np.isin(
            (node_us + uv_span) * (2 * uv_span + 1) + (node_vs + uv_span),
            np.array(
                [
                    (int(hex.u) + uv_span) * (2 * uv_span + 1) + (int(hex.v) + uv_span)
                    for hex in hexes
                ],
                dtype=np.int64,
            ),
        ),
    )[0].astype(np.int64)


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
    List[int], List[int], List[np.ndarray], List[float], List[int], List[int], List[str], int,
]:
    entry_bs, entry_nodes, entry_gts, entry_cost_scales, entry_cost_t0s, entry_cost_pd_nds, entry_part_keys = (
        [], [], [], [], [], [], [],
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
                t0 = t0_by_hex.get(hex_idx, 0)
                for node in nodes:
                    entry_bs.append(b)
                    entry_nodes.append(int(node))
                    if gt_trace is not None:
                        entry_gts.append(gt_trace)
                    entry_cost_scales.append(1.0)
                    if waveform_mse:
                        entry_cost_t0s.append(t0)
                        entry_cost_pd_nds.append(pd_nd_idx)
                        entry_part_keys.append(
                            cell_part_key(spec.contrast, cell, PD_ND_LABELS[pd_nd_idx])
                        )
    return (
        entry_bs, entry_nodes, entry_gts, entry_cost_scales,
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
    hexes = cost_sti_hexes(connectome, cost_radius=cost_radius)
    cost_hex_idxs = [hex_idx[(int(hex.u), int(hex.v))] for hex in hexes]
    hex_by_idx = {hex_idx: hex for hex, hex_idx in zip(hexes, cost_hex_idxs)}

    def _nodes_from_hex_type(b, hex_idx, cell):
        hex = hex_by_idx[hex_idx]
        return connectome.nodes_at_uv(hex.u, hex.v, cell, cells=cells)

    (
        entry_bs, entry_nodes, entry_gts, entry_cost_scales,
        entry_cost_t0s, entry_cost_pd_nds, entry_part_keys,
        _,
    ) = _assemble_mbar_readouts(
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

    if not entry_bs:
        raise ValueError("no moving-bar cost nodes (check subtypes and sti hexes)")

    gts, cost_scales, entry_bs, entry_nodes, cost_t0s, cost_pd_nds, power = (
        _pack_mbar_entries(
            entry_bs, entry_nodes, entry_gts, entry_cost_scales,
            entry_cost_t0s, entry_cost_pd_nds,
            device=device, sim_dtype=sim_dtype, waveform_mse=waveform_mse,
        )
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
    hexes = cost_sti_hexes(connectome, cost_radius=cost_radius)
    if at_x is not None or at_y is not None:
        filt_hexes = sti_hexes_at_xy(hexes, at_x=at_x, at_y=at_y)
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
    pack = MbarPack(
        task="mbar",
        contrast=contrast,
        i_sti=mbar_gt.i_sti,
        gts=mbar_gt.gts,
        cost_scales=mbar_gt.cost_scales,
        entry_bs=mbar_gt.entry_bs,
        entry_nodes=mbar_gt.entry_nodes,
        cost_t0s=mbar_gt.cost_t0s,
        cost_radius=cost_radius,
        cost_pd_nds=mbar_gt.cost_pd_nds,
        entry_part_keys=mbar_gt.entry_part_keys,
        cost_part_plot_specs=(
            _mbar_cost_part_plot_specs(
                mbar_gt.entry_part_keys,
                mbar_gt.entry_nodes,
                mbar_gt.cost_scales,
                mbar_gt.cost_pd_nds,
                connectome,
                contrast,
            )
            if mbar_gt.cost_pd_nds is not None
            else None
        ),
    )
    return pack, sti_opts, (
        f"moving-bar {contrast} (B={mbar_gt.n_b} stis, "
        f"{int(mbar_gt.entry_bs.shape[0])} cost nodes, "
        f"{cost_hex_label(cost_radius, mbar_gt.n_cost_hex)})"
    )
