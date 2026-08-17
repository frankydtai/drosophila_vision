# -*- coding: utf-8 -*-
"""Spread pack: uniform ``i_sti`` over all sti nodes; ir-only gt cost."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from neuron.borst import t_from_ms
from task.spread.gt import (
    GT_CELLS,
    RF_SIGN,
    contrast_sign,
    expand_gt_cells,
    load_ir,
    spread_gt_active,
)
from task.spread.sti_spec import sti_mask


@dataclass
class SpreadGt:
    i_sti: torch.Tensor
    gts: torch.Tensor
    entry_bs: torch.Tensor
    entry_nodes: torch.Tensor


def build_spread_gt(
    connectome,
    *,
    n_t: int,
    t_onset: int,
    i_baseline: float,
    i_sti: float,
    contrast: str,
    gt_amp: float,
    delta_ms: float,
    device: Optional[str] = None,
    sim_dtype: torch.dtype,
    ms_sti: Optional[float] = None,
    ms_response: Optional[float] = None,
    gt_cells: Optional[Sequence[str]] = None,
    filter: str = "none",
    spread_gt_mode: str = "all",
) -> SpreadGt:
    device = device or connectome.device
    if ms_response is None:
        raise ValueError("build_spread_gt requires ms_response")
    n_t_gt = int(t_onset) + t_from_ms(float(ms_response), delta_ms=float(delta_ms)) + 1
    if n_t_gt > int(n_t):
        raise ValueError(
            f"spread gt n_t={n_t_gt} exceeds forward n_t={n_t} "
            f"(ms_response={ms_response:g}, t_onset={t_onset})"
        )
    ir = load_ir(
        t_onset=t_onset,
        n_t=n_t_gt,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        filter=filter,
    )
    cell_idx = {cell: i for i, cell in enumerate(GT_CELLS)}
    gt_cells = [
        cell
        for cell in (GT_CELLS if gt_cells is None else expand_gt_cells(gt_cells))
        if cell in connectome.cells
    ]
    if not gt_cells:
        raise ValueError(f"spread has no gt cells (requested subset of {list(GT_CELLS)!r})")
    connectome_cell_idx = {cell: i for i, cell in enumerate(connectome.cells)}
    node_cell_idx = connectome.node_cells.detach().cpu().numpy()
    nodes_by_cell = {
        cell: np.flatnonzero(node_cell_idx == connectome_cell_idx[cell])
        for cell in gt_cells
    }
    i_sti_pulse = torch.as_tensor(
        (float(i_sti) - i_baseline) * sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms),
        dtype=sim_dtype,
        device=device,
    )
    sti_nodes = torch.as_tensor(connectome.sti_nodes, dtype=torch.long, device=device)
    i_sti = torch.zeros((1, n_t, connectome.n_node), dtype=sim_dtype, device=device)
    if len(sti_nodes):
        i_sti[:, :, sti_nodes] = float(i_baseline) + i_sti_pulse[:, None]
    entry_nodes = []
    gts = []
    for cell in gt_cells:
        if not spread_gt_active(spread_gt_mode, contrast, int(RF_SIGN[cell])):
            continue
        gt = ir[cell_idx[cell]][slice(t_onset, n_t_gt)] * gt_amp * float(contrast_sign(contrast))
        for node in nodes_by_cell[cell]:
            entry_nodes.append(int(node))
            gts.append(gt)
    if not entry_nodes:
        raise ValueError("no spread cost nodes (check gt cells)")
    gts = torch.tensor(np.asarray(gts), dtype=sim_dtype, device=device)
    entry_nodes = torch.tensor(entry_nodes, dtype=torch.long, device=device)
    return SpreadGt(
        i_sti=i_sti,
        gts=gts,
        entry_bs=torch.zeros(len(entry_nodes), dtype=torch.long, device=device),
        entry_nodes=entry_nodes,
    )
