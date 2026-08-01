# -*- coding: utf-8 -*-
"""Cost assembly, gradient descent, and multi-run driver.

Consumes a :class:`~training.readout_pack.TrainSession` and the model forward
(``neuron.run_full`` + ``neuron.readout``); produces per-part
unweighted costs, the weighted total, and the Adam training loop.

Traces are ``v`` (``v - v_onset``); cost multiplies model traces by ``out_scale``.

Sparse cost time points (#4): ``pack.data`` stays full post-onset length and the
subsample is gathered from both model trace and gt at cost time via
``pack.cost_time_ix``; ``power`` is recomputed on the subsample.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from training.readout_pack import SIM_DTYPE
from neuron import run_full
from neuron.readout import (
    CA_PACK_READOUTS,
    pack_needs_waveform_mse,
    window_time_traces,
)

from training.config import (
    MOVING_BAR_TASKS,
    ND_IDX,
    PD_IDX,
    PD_ND_LABELS,
    cost_part_keys_for_readout,
    expand_cost_weight_dict,
    moving_bar_cost_part_key,
    session_cost_part_keys,
)
from training.params import params_from_z, schema_bounds, schema_guess, schema_nparams
from training.readout_pack import (
    FusedForward,
    ModelBackend,
    ReadoutPack,
    TrainingResult,
    TrainSession,
    active_device,
)

from task.moving_bar.data import (
    bar_specs_for_session,
    cost_dsi_from_sel,
    dsi_sequential_batch_pairs,
    remap_dsi_rows,
)


def ca_cost(ca, data, session: TrainSession, scale=1.0, power=None):
    if power is None:
        power = session.primary_readout.power
    pack = session.primary_readout
    pack_t_onset = int(pack.signal.shape[1] - pack.data.shape[1])
    mt = session.n_t
    return torch.sum((scale * ca - data[pack_t_onset:mt])**2) / power * 100.0


def out_scale_for_nodes(p, node_index, backend: ModelBackend, *, sim_dtype=SIM_DTYPE):
    """Per-node ``out_scale`` using the same indexing as ``_pack_out_scale``."""
    os_param = p.get('out_scale', 1.0)
    n = int(node_index.shape[0])
    dev = node_index.device
    if not torch.is_tensor(os_param) or os_param.dim() == 0:
        val = float(os_param if not torch.is_tensor(os_param) else os_param.item())
        return torch.full((n,), val, dtype=sim_dtype, device=dev)
    if backend.network is not None:
        ci = backend.network.node_cell[node_index]
    else:
        ci = node_index % backend.n_cells
    return os_param[ci]


def _pack_out_scale(p, pack: ReadoutPack, backend: ModelBackend, session: TrainSession):
    """Per-cost-row output scale from schema ``out_scale``."""
    return out_scale_for_nodes(
        p, pack.readout_node, backend, sim_dtype=session.sim_dtype,
    )


def _pack_ca_readouts(p, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    try:
        readout = CA_PACK_READOUTS[session.model]
    except KeyError:
        raise ValueError(f"no pack readout for model={session.model!r}") from None
    return readout(p, pack, session, batch_idx)


def _subgroup_power(weight, data):
    power = torch.sum(weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=data.dtype, device=data.device)
    return power


def _pack_cost_mse(scale, data, weight, sel, power):
    diff = scale[:, None] * sel - data
    return torch.sum(weight[:, None] * diff ** 2) / power * 100.0


def _part_weight(session: TrainSession, part_key: str) -> float:
    return float(session.cost_weights.get(part_key, 1.0))


def _pack_part_key_for_cell(pack: ReadoutPack, cell_idx: int) -> str:
    if pack.cost_pd_nd is not None:
        label = PD_ND_LABELS[int(pack.cost_pd_nd[cell_idx].item())]
        return moving_bar_cost_part_key(pack.name, label)
    return pack.name


def _pack_has_active_cost(pack: ReadoutPack, session: TrainSession) -> bool:
    for key in cost_part_keys_for_readout(pack.name):
        if _part_weight(session, key) != 0.0:
            return True
    return False


def _pack_has_active_mse(pack: ReadoutPack, session: TrainSession) -> bool:
    """True if waveform MSE parts (pack name or PD/ND) have non-zero weight."""
    if pack.name in MOVING_BAR_TASKS:
        return any(
            _part_weight(session, moving_bar_cost_part_key(pack.name, lab)) != 0.0
            for lab in PD_ND_LABELS
        )
    return _part_weight(session, pack.name) != 0.0


def _mse_active_row_mask(pack: ReadoutPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost rows with non-zero PD/ND (or pack) weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    if pack.name in MOVING_BAR_TASKS:
        if not _pack_has_active_mse(pack, session) or pack.cost_pd_nd is None:
            return torch.zeros(n, dtype=torch.bool, device=dev)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        for idx, label in ((PD_IDX, "PD"), (ND_IDX, "ND")):
            if _part_weight(session, moving_bar_cost_part_key(pack.name, label)) != 0.0:
                mask |= pack.cost_pd_nd == int(idx)
        return mask
    on = _part_weight(session, pack.name) != 0.0
    return torch.full((n,), on, dtype=torch.bool, device=dev)


def _dsi_active_row_mask(pack: ReadoutPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost rows needed by a non-zero DSI weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
    if (
        pack.dsi_pos_rows is None
        or pack.dsi_pos_rows.numel() == 0
        or _part_weight(session, dsi_key) == 0.0
    ):
        return mask
    mask[pack.dsi_pos_rows] = True
    mask[pack.dsi_neg_rows] = True
    return mask


def _pack_active_batch_indices(pack: ReadoutPack, session: TrainSession) -> Tuple[int, ...]:
    """Stimulus batch indices with at least one non-zero-weight cost node."""
    row_mask = _mse_active_row_mask(pack, session) | _dsi_active_row_mask(pack, session)
    if not bool(row_mask.any()):
        return ()
    batches = pack.readout_batch[row_mask].unique(sorted=True)
    return tuple(int(b) for b in batches.tolist())


def _active_row_indices(
    pack: ReadoutPack,
    session: TrainSession,
    batch_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    keep = _mse_active_row_mask(pack, session) | _dsi_active_row_mask(pack, session)
    if batch_idx is not None:
        keep = keep & (pack.readout_batch == int(batch_idx))
    if not bool(keep.any()):
        return None
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def _slice_pack_rows(pack: ReadoutPack, row_ix: torch.Tensor) -> ReadoutPack:
    fields = {
        "data": pack.data[row_ix],
        "cost_weight": pack.cost_weight[row_ix],
        "readout_batch": pack.readout_batch[row_ix],
        "readout_node": pack.readout_node[row_ix],
    }
    if pack.cost_t0 is not None:
        fields["cost_t0"] = pack.cost_t0[row_ix]
    if pack.cost_radius is not None:
        fields["cost_radius"] = pack.cost_radius[row_ix]
    if pack.readout_stim_u is not None:
        fields["readout_stim_u"] = pack.readout_stim_u[row_ix]
    if pack.readout_stim_v is not None:
        fields["readout_stim_v"] = pack.readout_stim_v[row_ix]
    if pack.cost_pd_nd is not None:
        fields["cost_pd_nd"] = pack.cost_pd_nd[row_ix]
    fields.update(remap_dsi_rows(pack, row_ix))
    return replace(pack, **fields)


def _subset_pack_batches(pack: ReadoutPack, batch_indices: Tuple[int, ...]) -> Optional[ReadoutPack]:
    if len(batch_indices) == int(pack.signal.shape[0]):
        return pack
    dev = pack.signal.device
    idx_t = torch.tensor(batch_indices, dtype=torch.long, device=dev)
    rb = pack.readout_batch
    keep = torch.isin(rb, idx_t)
    if not bool(keep.any()):
        return None
    lut_size = int(max(max(batch_indices), int(rb.max()))) + 1
    lut = torch.full((lut_size,), -1, dtype=torch.long, device=dev)
    lut[idx_t] = torch.arange(len(batch_indices), dtype=torch.long, device=dev)
    new_rb = lut[rb[keep]]
    kept_old = torch.nonzero(keep, as_tuple=False).reshape(-1)
    fields = {
        "signal": pack.signal.index_select(0, idx_t),
        "data": pack.data[keep],
        "cost_weight": pack.cost_weight[keep],
        "readout_batch": new_rb,
        "readout_node": pack.readout_node[keep],
    }
    if pack.cost_t0 is not None:
        fields["cost_t0"] = pack.cost_t0[keep]
    if pack.cost_radius is not None:
        fields["cost_radius"] = pack.cost_radius[keep]
    if pack.readout_stim_u is not None:
        fields["readout_stim_u"] = pack.readout_stim_u[keep]
    if pack.readout_stim_v is not None:
        fields["readout_stim_v"] = pack.readout_stim_v[keep]
    if pack.cost_pd_nd is not None:
        fields["cost_pd_nd"] = pack.cost_pd_nd[keep]
    fields.update(remap_dsi_rows(pack, kept_old))
    return replace(pack, **fields)


def _pack_for_active_cost(
    pack: ReadoutPack,
    session: TrainSession,
    *,
    batch_idx: Optional[int] = None,
    batch_indices: Optional[Tuple[int, ...]] = None,
) -> Optional[ReadoutPack]:
    """Drop zero-weight rows and, when requested, inactive stimulus batches."""
    work = pack
    if batch_indices is not None:
        work = _subset_pack_batches(pack, batch_indices)
        if work is None:
            return None
    rows = _active_row_indices(work, session, batch_idx=batch_idx)
    if rows is None:
        return None
    return _slice_pack_rows(work, rows)


def _build_cost_subpacks(session: TrainSession) -> Dict[str, ReadoutPack]:
    """Active cost row/batch subsets per task (batched mode only)."""
    if session.sequential:
        return {}
    out: Dict[str, ReadoutPack] = {}
    for name, pack in session.readouts.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
        if sub is not None:
            out[name] = sub
    return out


def _signal_fuse_key(pack: ReadoutPack) -> Tuple:
    """Group packs that can share one ``run_full`` (shape, scale, onset)."""
    sig = pack.signal
    t_onset = int(sig.shape[1] - pack.data.shape[1])
    return (
        int(sig.shape[1]),
        int(sig.shape[2]),
        str(sig.device),
        sig.dtype,
        float(pack.signal_scale),
        t_onset,
    )


def _build_fused_forward(
    session: TrainSession,
    cost_subpacks: Dict[str, ReadoutPack],
) -> Tuple[FusedForward, ...]:
    if session.sequential or not cost_subpacks:
        return ()
    by_key: Dict[Tuple, List[ReadoutPack]] = {}
    for pack in cost_subpacks.values():
        by_key.setdefault(_signal_fuse_key(pack), []).append(pack)
    fused: List[FusedForward] = []
    for packs in by_key.values():
        offsets: List[int] = []
        off = 0
        for pack in packs:
            offsets.append(off)
            off += int(pack.signal.shape[0])
        fused.append(FusedForward(subpacks=tuple(packs), batch_offsets=tuple(offsets)))
    return tuple(fused)


def _readout_from_trace_full(
    trace_full: torch.Tensor,
    pack: ReadoutPack,
    *,
    batch_offset: int = 0,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    rb = pack.readout_batch if batch_offset == 0 else pack.readout_batch + batch_offset
    pack_t_onset = int(pack.signal.shape[1] - pack.data.shape[1])
    dsi_sel = trace_full[rb, pack_t_onset:, pack.readout_node]
    if not pack_needs_waveform_mse(pack):
        return None, dsi_sel
    if pack.cost_t0 is None:
        return dsi_sel, dsi_sel
    mse_sel = window_time_traces(
        trace_full, rb, pack.readout_node, pack.cost_t0,
        win=pack.data.shape[1], t_onset=pack_t_onset,
    )
    return mse_sel, dsi_sel


def _pack_cost_dsi_from_sel(
    pack: ReadoutPack,
    session: TrainSession,
    scale: torch.Tensor,
    dsi_sel: torch.Tensor,
) -> Optional[torch.Tensor]:
    """DSI cost from full post-stimulus traces; independent of cost windows."""
    key = moving_bar_cost_part_key(pack.name, "DSI")
    if _part_weight(session, key) == 0.0:
        return None
    return cost_dsi_from_sel(pack, scale, dsi_sel)


def _pack_cost_parts_from_sel(
    pack: ReadoutPack,
    session: TrainSession,
    scale: torch.Tensor,
    sel: Optional[torch.Tensor],
    dsi_sel: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pd_nd = pack.cost_pd_nd
    if pack.name in MOVING_BAR_TASKS:
        out: Dict[str, torch.Tensor] = {}
        if pd_nd is not None:
            for pd_nd_idx, label in ((PD_IDX, "PD"), (ND_IDX, "ND")):
                key = moving_bar_cost_part_key(pack.name, label)
                if _part_weight(session, key) == 0.0:
                    continue
                if sel is None:
                    raise ValueError(
                        f"waveform readout required for {key} but pack has no cost window",
                    )
                mask = pd_nd == pd_nd_idx
                if not bool(mask.any()):
                    out[key] = torch.zeros(
                        (), dtype=session.sim_dtype, device=session.device,
                    )
                    continue
                power = _subgroup_power(pack.cost_weight[mask], pack.data[mask])
                out[key] = _pack_cost_mse(
                    scale[mask], pack.data[mask], pack.cost_weight[mask],
                    sel[mask], power,
                )
        dsi_part = _pack_cost_dsi_from_sel(pack, session, scale, dsi_sel)
        if dsi_part is not None:
            out[moving_bar_cost_part_key(pack.name, "DSI")] = dsi_part
        return out
    if sel is None:
        raise ValueError(f"waveform readout required for pack {pack.name!r}")
    data = pack.data
    weight = pack.cost_weight
    power = pack.power
    # #4 sparse time points: gather model + gt on the requested post-onset
    # t indices and recompute power over the subsample.
    if pack.cost_time_ix is not None:
        ix = pack.cost_time_ix.to(device=sel.device)
        sel = sel.index_select(1, ix)
        data = data.index_select(1, ix)
        power = _subgroup_power(weight, data)
    return {pack.name: _pack_cost_mse(scale, data, weight, sel, power)}


def _pack_cost_parts_from_fused_trace(
    p,
    pack: ReadoutPack,
    session: TrainSession,
    trace_full: torch.Tensor,
    *,
    batch_offset: int = 0,
) -> Dict[str, torch.Tensor]:
    scale = _pack_out_scale(p, pack, session.backend, session)
    sel, dsi_sel = _readout_from_trace_full(
        trace_full, pack, batch_offset=batch_offset,
    )
    return _pack_cost_parts_from_sel(pack, session, scale, sel, dsi_sel)


def _calc_cost_parts_fused(
    p,
    session: TrainSession,
) -> Dict[str, torch.Tensor]:
    parts: Dict[str, torch.Tensor] = {}
    for group in session.fused_forward:
        if len(group.subpacks) == 1:
            sig = group.subpacks[0].signal
        else:
            sig = torch.cat([pack.signal for pack in group.subpacks], dim=0)
        # Same fuse key ⇒ shared signal_scale / t_onset; pass one subpack for prepare.
        trace_full = run_full(session, p, sig, pack=group.subpacks[0])
        for pack, off in zip(group.subpacks, group.batch_offsets):
            for key, part in _pack_cost_parts_from_fused_trace(
                p, pack, session, trace_full, batch_offset=off,
            ).items():
                if _part_weight(session, key) != 0.0:
                    parts[key] = part
    return parts


def _pack_cost_forward(p, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    scale = _pack_out_scale(p, pack, session.backend, session)
    pd_nd = pack.cost_pd_nd
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        if not bool(mask.any()):
            return None
        scale = scale[mask]
        data = pack.data[mask]
        weight = pack.cost_weight[mask]
        if pd_nd is not None:
            pd_nd = pd_nd[mask]
    else:
        data = pack.data
        weight = pack.cost_weight
    sel, dsi_sel = _pack_ca_readouts(p, pack, session, batch_idx)
    return scale, data, weight, sel, dsi_sel, pd_nd


def _pack_cost_parts_from_params(p, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    """Unweighted cost parts for one pack (PD/ND split for moving_bar)."""
    fwd = _pack_cost_forward(p, pack, session, batch_idx)
    if fwd is None:
        return {}
    scale, data, weight, sel, dsi_sel, pd_nd = fwd
    return _pack_cost_parts_from_sel(pack, session, scale, sel, dsi_sel)


def _pack_cost_rows(p, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    """Forward + MSE for one pack (full aggregate; used for diagnostics)."""
    fwd = _pack_cost_forward(p, pack, session, batch_idx)
    if fwd is None:
        return None
    scale, data, weight, sel, dsi_sel, _pd_nd = fwd
    if sel is None:
        return _pack_cost_dsi_from_sel(pack, session, scale, dsi_sel)
    return _pack_cost_mse(scale, data, weight, sel, pack.power)


def _pack_cost_parts_for_pack(z, pack: ReadoutPack, session: TrainSession, batch_idx=None, p=None):
    if p is None:
        p = params_from_z(z, session)
    return _pack_cost_parts_from_params(p, pack, session, batch_idx)


def _pack_cost_part(z, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    parts = _pack_cost_parts_for_pack(z, pack, session, batch_idx)
    if not parts:
        return torch.zeros((), dtype=session.sim_dtype, device=session.device)
    return sum(parts.values())


def _pack_cost(z, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    return _pack_cost_part(z, pack, session, batch_idx)


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-part unweighted cost (before ``cost_weights``)."""
    p = params_from_z(z, session)
    if session.fused_forward:
        return _calc_cost_parts_fused(p, session)
    parts: Dict[str, torch.Tensor] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    if session.cost_subpacks and not session.sequential:
        for _name, pack in session.cost_subpacks.items():
            if not _pack_has_active_cost(pack, session):
                continue
            pack_parts = _pack_cost_parts_from_params(p, pack, session, batch_idx=None)
            for part_key, part in pack_parts.items():
                if _part_weight(session, part_key) == 0.0:
                    continue
                parts[part_key] = part
        return parts
    for _name, pack in session.readouts.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        if session.sequential:
            pack_parts: Dict[str, torch.Tensor] = {}
            if _pack_has_active_mse(pack, session):
                for b in active_batches:
                    sub = _pack_for_active_cost(pack, session, batch_idx=b)
                    if sub is None:
                        continue
                    for key, part in _pack_cost_parts_from_params(
                        p, sub, session, batch_idx=b,
                    ).items():
                        pack_parts[key] = pack_parts.get(key, zero) + part
            dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
            if _part_weight(session, dsi_key) != 0.0:
                for group in _dsi_sequential_batch_groups(pack, session):
                    sub_dsi = _pack_for_dsi_batch_group(pack, session, group)
                    if sub_dsi is None:
                        continue
                    dsi_parts = _pack_cost_parts_from_params(
                        p, sub_dsi, session, batch_idx=None,
                    )
                    if dsi_key in dsi_parts:
                        pack_parts[dsi_key] = (
                            pack_parts.get(dsi_key, zero) + dsi_parts[dsi_key]
                        )
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is None:
                continue
            pack_parts = _pack_cost_parts_from_params(p, sub, session, batch_idx=None)
        for part_key, part in pack_parts.items():
            if _part_weight(session, part_key) == 0.0:
                continue
            parts[part_key] = part
    return parts


def _weighted_cost_from_parts(parts: Dict[str, torch.Tensor], session: TrainSession):
    total = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    for name, part in parts.items():
        w = float(session.cost_weights.get(name, 1.0))
        total = total + w * part
    return total


def calc_cost(z, session: TrainSession):
    return _weighted_cost_from_parts(calc_cost_parts(z, session), session)


def _params_from_z(z, session: TrainSession):
    return params_from_z(z, session)


def _pack_spec_names(session: TrainSession, pack: ReadoutPack) -> Tuple[str, ...]:
    opts = ((session.train_opts or {}).get(f"{pack.name}_stimulus_opts")) or {}
    names = opts.get("spec_names")
    if names:
        return tuple(str(s) for s in names)
    return tuple(s.name for s in bar_specs_for_session(session, pack.name))


def _dsi_sequential_batch_groups(
    pack: ReadoutPack, session: TrainSession,
) -> Tuple[Tuple[int, ...], ...]:
    """Active DSI microbatches: each group is one axis x width (typically B=2)."""
    active = set(_pack_active_batch_indices(pack, session))
    groups: list[tuple[int, ...]] = []
    for pair in dsi_sequential_batch_pairs(_pack_spec_names(session, pack)):
        kept = tuple(b for b in pair if b in active)
        if len(kept) < 2:
            continue
        groups.append(kept)
    return tuple(groups)


def _pack_for_dsi_batch_group(
    pack: ReadoutPack,
    session: TrainSession,
    batch_indices: Tuple[int, ...],
) -> Optional[ReadoutPack]:
    """Subset to one DSI direction pair; keep parent ``dsi_power`` for additive costs."""
    sub = _pack_for_active_cost(pack, session, batch_indices=batch_indices)
    if sub is None or pack.dsi_power is None:
        return sub
    return replace(sub, dsi_power=pack.dsi_power)


def _iter_cost_microbatches(session: TrainSession):
    """Yield ``(pack, batch_idx, sub_pack)`` for gradient accumulation."""
    for _name, pack in session.readouts.items():
        if not _pack_has_active_cost(pack, session):
            continue
        active_batches = _pack_active_batch_indices(pack, session)
        if not active_batches:
            continue
        if session.sequential:
            if _pack_has_active_mse(pack, session):
                for b in active_batches:
                    sub = _pack_for_active_cost(pack, session, batch_idx=b)
                    if sub is not None:
                        yield pack, b, sub
            dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
            if _part_weight(session, dsi_key) != 0.0:
                for group in _dsi_sequential_batch_groups(pack, session):
                    sub_dsi = _pack_for_dsi_batch_group(pack, session, group)
                    if sub_dsi is not None:
                        yield pack, None, sub_dsi
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is not None:
                yield pack, None, sub


def backward_accum_weighted_cost(z, session: TrainSession):
    """Backward weighted cost one micro-batch at a time (releases graph each step)."""
    parts_sum: Dict[str, float] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    for pack, batch_idx, sub in _iter_cost_microbatches(session):
        p = _params_from_z(z, session)
        mb_loss = zero
        has_loss = False
        dsi_only = (
            session.sequential
            and batch_idx is None
            and pack.name in MOVING_BAR_TASKS
        )
        dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
        for key, part in _pack_cost_parts_from_params(
            p, sub, session, batch_idx=batch_idx,
        ).items():
            if dsi_only and key != dsi_key:
                continue
            if (not dsi_only) and session.sequential and key == dsi_key:
                # single-batch slices have no complete DSI pairs; skip zeros
                continue
            w = _part_weight(session, key)
            if w == 0.0:
                continue
            mb_loss = mb_loss + w * part
            has_loss = True
            parts_sum[key] = parts_sum.get(key, 0.0) + float(part.item())
        if has_loss:
            mb_loss.backward()
    total = sum(_part_weight(session, k) * v for k, v in parts_sum.items())
    return total, parts_sum


def _float_parts_dict(parts: Optional[Dict[str, torch.Tensor]], task_order=None):
    if not parts:
        return None
    out = {k: float(v.item() if torch.is_tensor(v) else v) for k, v in parts.items()}
    if task_order:
        return {k: out[k] for k in task_order if k in out}
    return out


def _fmt_cost_parts(parts):
    if not parts:
        return ""
    return "  [" + "  ".join(f"{k}={v:.4f}" for k, v in parts.items()) + "]"


_TQDM_REFRESH_INTERVAL = 10


def gradient_network(z, lr=0.0001, cost_fn=None, n_steps=100, device="cpu", z_bounds=None,
                     cost_log=None, step_log=None, float_last_parts=None, task_order=None,
                     backward_step=None, eval_cost=None,
                     checkpoint_interval=None, on_interval_best=None, global_step_start=0):

    a = time.time()

    z = nn.Parameter(z.clone().to(device))

    optimizer = torch.optim.Adam([z], lr=lr)

    def _measure_cost(param_z):
        if eval_cost is not None:
            return eval_cost(param_z)
        return cost_fn(param_z).item()

    try:
        cost = _measure_cost(z)
    except RuntimeError as e:
        raise RuntimeError(f'non-finite at init: {e}') from e
    if not np.isfinite(cost):
        raise RuntimeError(f'non-finite cost at init: {cost}')
    best_cost = cost
    best_z = z.clone().detach()
    interval_best_cost = cost
    interval_best_z = z.clone().detach()

    initial_cost = 1.0 * cost
    initial_parts = float_last_parts(task_order) if float_last_parts else None
    best_parts = initial_parts

    def _reset_interval_from_z():
        nonlocal interval_best_cost, interval_best_z
        interval_best_cost = _measure_cost(z)
        interval_best_z = z.clone().detach()

    def _commit_interval_checkpoint(global_step):
        nonlocal optimizer
        if on_interval_best is not None:
            on_interval_best(global_step, interval_best_z, interval_best_cost)
        with torch.no_grad():
            z.copy_(interval_best_z)
        optimizer = torch.optim.Adam([z], lr=lr)
        _reset_interval_from_z()

    progress_bar = tqdm(
        range(n_steps),
        desc=f'Cost: {cost:.4f}' + _fmt_cost_parts(initial_parts),
        miniters=_TQDM_REFRESH_INTERVAL,
        maxinterval=60,
        file=sys.stderr,
    )
    aborted = None

    for i in progress_bar:

        optimizer.zero_grad()

        try:
            if backward_step is not None:
                cost = backward_step(z)
            else:
                cost_t = cost_fn(z)
                cost = cost_t.item()
                cost_t.backward()
        except RuntimeError as e:
            aborted = f'step {i}: {e}'
            break

        if not np.isfinite(cost):
            aborted = f'step {i}: non-finite cost={cost}'
            break
        if not torch.isfinite(z).all():
            aborted = f'step {i}: non-finite z'
            break
        if z.grad is not None and not torch.isfinite(z.grad).all():
            aborted = f'step {i}: non-finite grad'
            break

        if cost < best_cost:

            best_cost = cost
            best_z = z.clone().detach()
            if float_last_parts is not None:
                best_parts = float_last_parts(task_order)

        if cost < interval_best_cost:
            interval_best_cost = cost
            interval_best_z = z.clone().detach()

        if cost_log is not None:
            cost_log.append(cost)
        if step_log is not None:
            step_log(z)

        optimizer.step()

        with torch.no_grad():

            z.clamp_(z_bounds[:, 0].to(device), z_bounds[:, 1].to(device))

        global_step = global_step_start + i + 1
        if checkpoint_interval and global_step % checkpoint_interval == 0:
            _commit_interval_checkpoint(global_step)

        step_parts = float_last_parts(task_order) if float_last_parts else None
        if (i + 1) % _TQDM_REFRESH_INTERVAL == 0 or i == n_steps - 1:
            progress_bar.set_description(
                f'Cost: {cost:.4f}' + _fmt_cost_parts(step_parts),
                refresh=False,
            )

    if aborted is None:
        try:
            cost = _measure_cost(z)
            final_parts = float_last_parts(task_order) if float_last_parts else None
        except RuntimeError as e:
            aborted = f'final eval: {e}'
            cost = float('nan')
            final_parts = None
        else:
            if np.isfinite(cost) and cost < best_cost:
                best_cost = cost
                best_z = z.clone().detach()
                best_parts = final_parts
    else:
        cost = float('nan')
        final_parts = None

    print()
    if aborted is not None:
        print('ABORT:', aborted)
    print('Initl cost =', format(initial_cost, '.4f') + _fmt_cost_parts(initial_parts))
    print('Final cost =', format(cost, '.4f') + _fmt_cost_parts(final_parts))
    print('Best  cost =', format(best_cost, '.4f') + _fmt_cost_parts(best_parts))

    b = time.time()

    print('time needed  =', format(b - a, '.2f'), ' sec')
    print()

    return best_z


def train_staged(z, cost_fn, z_bounds, lrs, nsteps, cost_log=None, step_log=None,
                 float_last_parts=None, task_order=None,
                 backward_step=None, eval_cost=None,
                 checkpoint_interval=None, on_interval_best=None, global_step_start=0):
    # run gradient_network once per learning-rate stage, chaining the best params.
    global_step = global_step_start
    for lr in lrs:
        z = gradient_network(z, lr=lr, n_steps=nsteps, device=active_device(),
                             cost_fn=cost_fn, z_bounds=z_bounds, cost_log=cost_log,
                             step_log=step_log, float_last_parts=float_last_parts,
                             task_order=task_order,
                             backward_step=backward_step, eval_cost=eval_cost,
                             checkpoint_interval=checkpoint_interval,
                             on_interval_best=on_interval_best,
                             global_step_start=global_step)
        global_step += nsteps
    return z


def _make_step_logger(session: TrainSession):
    """Build training step hooks for :func:`gradient_network`."""
    part_keys = session_cost_part_keys(session.task_list)
    target_history = {name: [] for name in part_keys}
    _last_parts: Optional[Dict[str, float]] = None
    _last_total: Optional[float] = None

    def _set_last(parts, total):
        nonlocal _last_parts, _last_total
        _last_parts = dict(parts)
        _last_total = float(total)

    def cost_fn(z):
        parts = calc_cost_parts(z, session)
        total = _weighted_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return total

    def eval_cost(z):
        with torch.no_grad():
            parts = calc_cost_parts(z, session)
            total = _weighted_cost_from_parts(parts, session)
        _set_last({k: float(v.item()) for k, v in parts.items()}, float(total.item()))
        return float(total.item())

    def backward_step(z):
        total, parts_sum = backward_accum_weighted_cost(z, session)
        _set_last(parts_sum, total)
        return total

    def log_step(z=None):
        if _last_parts is None or _last_total is None:
            raise RuntimeError("log_step called before cost_fn in the same training step")
        for name in part_keys:
            if name in _last_parts:
                target_history[name].append(float(_last_parts[name]))
            else:
                target_history[name].append(0.0)
        return float(_last_total)

    def float_last_parts(task_order=None):
        if _last_parts is None:
            raise RuntimeError("float_last_parts called before cost_fn")
        return _float_parts_dict(_last_parts, task_order)

    if session.sequential:
        return cost_fn, target_history, log_step, float_last_parts, backward_step, eval_cost
    return cost_fn, target_history, log_step, float_last_parts, None, None


def do_many_runs(session: TrainSession, nofruns, nofsteps, lrs=(0.1, 0.01, 0.001),
                 z_init=None, checkpoint_interval=None, checkpoint_outdir=None,
                 make_checkpoint_callback=None, checkpoint_on_png=None) -> TrainingResult:
    """Run ``nofruns`` independent fits; return arrays (no file I/O)."""
    schema = list(session.schema)
    n_params = schema_nparams(schema)
    bounds = schema_bounds(schema, session.sim_dtype)

    all_params = np.zeros((nofruns, n_params))
    final_costs = np.zeros(nofruns)
    part_keys = session_cost_part_keys(session.task_list)
    final_costs_by_task = {name: np.zeros(nofruns) for name in part_keys}
    best_i = 0
    best_cost = np.inf
    cost_curve = np.array([], dtype=np.float64)
    cost_curves_by_task = {}

    for i in range(nofruns):
        print()
        print('round', i)
        print()

        z = z_init.clone() if z_init is not None else schema_guess(schema, session.sim_dtype)
        cost_history = []
        (cost_fn, target_history, log_step, float_last_parts,
         backward_step, eval_cost) = _make_step_logger(session)

        def step_log(z):
            cost_history.append(log_step(z))

        on_interval_best = None
        if checkpoint_interval is not None:
            if checkpoint_outdir is None or make_checkpoint_callback is None:
                raise ValueError(
                    "checkpoint_interval requires checkpoint_outdir and make_checkpoint_callback"
                )
            on_interval_best = make_checkpoint_callback(
                checkpoint_outdir, session, run_i=i, nofruns=nofruns,
                on_png=checkpoint_on_png,
            )

        z_fit = train_staged(
            z, cost_fn, bounds, lrs, nofsteps,
            step_log=step_log,
            float_last_parts=float_last_parts,
            task_order=list(part_keys),
            backward_step=backward_step,
            eval_cost=eval_cost,
            checkpoint_interval=checkpoint_interval,
            on_interval_best=on_interval_best,
        )

        all_params[i] = z_fit.detach().cpu().numpy()
        fit_parts = calc_cost_parts(z_fit, session)
        final_costs[i] = float(_weighted_cost_from_parts(fit_parts, session).item())
        for name, part in fit_parts.items():
            final_costs_by_task[name][i] = float(part.item())
        if final_costs[i] < best_cost:
            best_cost = final_costs[i]
            best_i = i
            cost_curve = np.array(cost_history, dtype=np.float64)
            cost_curves_by_task = {
                name: np.array(curve, dtype=np.float64)
                for name, curve in target_history.items()
            }

    return TrainingResult(
        all_params=all_params,
        final_costs=final_costs,
        best_i=best_i,
        cost_curve=cost_curve,
        cost_curves_by_task=cost_curves_by_task,
        final_costs_by_task=final_costs_by_task,
    )
