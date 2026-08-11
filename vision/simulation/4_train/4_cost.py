"""Cost assembly: per-part MSE / DSI and the weighted total.

Consumes a :class:`~train.session.TrainSession` and the model forward
(``neuron.forward`` + ``neuron.readout``); produces per-part local-% costs and
the mean weighted total (``Σ W·cost / Σ W``). The Adam / staged-lr loop lives in
:mod:`train.optimization`.

Owns cost-time execution plans (active subpacks / fused forward) built at
calc time — not at session open.

Readout traces are absolute ``v`` (``filter=none``) or ``ca`` (``filter=ca``);
cost compares the readout to ``gt_aff = a_gt * gt + bias_gt`` (``+ v_th`` when
present and not ``bias_gt_from_v_onset``). When ``train_opts['bias_gt_from_v_onset']``,
``params['bias_gt']`` is written from ``v`` at ``t_onset`` (or ``v_ca`` when ``filter=ca``),
clamped to schema ``bias_gt`` ``lo``/``hi``; ``bias_gt_from_v_onset_grad`` controls
whether that onset stays in the graph. Same for ``v_th_ca_from_v_th`` /
``a_ca_from_a_out``: sources are written into ``params['v_th_ca']`` / ``params['a_ca']``
(and ``param.csv``).
Waveform MSE normalization is ``session`` / ``train_opts`` ``cost_norm``:

* ``gt_power``: ``100 * Σ w (v_readout−gt_aff)² / Σ w (a_gt·gt)²``
  (no ``bias_gt`` / ``v_th`` in the denominator)
* ``a_gt2`` (default): ``Σ w (v_readout−gt_aff)² / a_gt²`` (per-entry ``a_i²``; bias not in denom)

**within each part**. The train total averages those part costs (equal
weight per cell×radius unless ``cost_weights`` says otherwise).

Sparse cost time points (#4): ``pack.gt`` stays the ``ms_response`` segment length
(spot excludes ``ms_post``) and the subsample is gathered from both v_readout
trace and gt at cost time via ``pack.cost_time_idx``; when radii use different
``cost_ms`` lists, ``pack.cost_time_mask`` zeros non-participating (entry, t)
pairs. ``gt_power`` is recomputed on the subsample.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neuron.forward import (
    ca_from_v_ca,
    forward_v,
    pack_t_onset,
    v_ca_from_v,
)
from neuron.readout import (
    pack_needs_waveform_mse,
    window_time_traces,
)

from train.config import (
    COST_NORMS,
    MOVING_BAR_TASKS,
    ND_IDX,
    PD_IDX,
    PD_ND_LABELS,
    coarse_weight_keys_for_part,
    cost_part_keys_for_readout,
    expand_cost_norm,
    moving_bar_cell_cost_part_key,
    moving_bar_cost_part_key,
    session_cost_part_keys,
    spot_cost_part_key,
)
from param_defaults import (
    COST_NORM as DEFAULT_COST_NORM,
    BIAS_GT_FROM_V_ONSET as DEFAULT_BIAS_GT_FROM_V_ONSET,
    FILTER as DEFAULT_FILTER,
)
from train.param import (
    ModelBackend,
    SIM_DTYPE,
    materialize_from_opts,
    params_from_z,
)
from train.session import ReadoutPack, TrainSession
from task.moving_bar.gt import (
    bar_specs_for_session,
    cost_dsi_from_v_readout_dsi,
    dsi_sequential_batch_sets,
    remap_dsi_entries,
)


@dataclass(frozen=True)
class FusedForward:
    """Packs with matching i_sti shape / onset; one ``forward_full`` per fuse."""

    subpacks: Tuple[ReadoutPack, ...]
    batch_offsets: Tuple[int, ...]


def pack_cost_abs_time_idx(pack: ReadoutPack, t_onset, *, spot_cost_radius=None):
    """Absolute time indices for sparse spot cost samples (or ``None``).

    Sole reader of ``cost_time_idx`` / ``cost_time_mask`` / ``spot_cost_radius``.
    ``spot_cost_radius`` is Euclidean; when set and a mask exists, keep that
    radius's columns only. Omit ``spot_cost_radius`` → union of all radii.
    """
    idx = pack.cost_time_idx
    if idx is None:
        return None
    base = int(t_onset or 0)
    idx_np = idx.detach().cpu().numpy().astype(np.int64, copy=False)
    if spot_cost_radius is None:
        return base + idx_np
    mask = pack.cost_time_mask
    rad_t = pack.spot_cost_radius
    if mask is None or rad_t is None:
        return base + idx_np
    rad = np.round(rad_t.detach().cpu().numpy().astype(float), 6)
    hit = np.where(rad == round(float(spot_cost_radius), 6))[0]
    if not hit.size:
        return base + np.zeros(0, dtype=np.int64)
    entry_mask = mask[int(hit[0])].detach().cpu().numpy() > 0
    return base + idx_np[entry_mask]


def _param_for_nodes(params, key: str, node_idx, backend: ModelBackend, *, sim_dtype=SIM_DTYPE):
    """Per-node values from a cell-indexed schema param (or scalar default)."""
    raw = params.get(key, 1.0 if key == "a_gt" else 0.0)
    n = int(node_idx.shape[0])
    dev = node_idx.device
    if not torch.is_tensor(raw) or raw.dim() == 0:
        val = float(raw if not torch.is_tensor(raw) else raw.item())
        return torch.full((n,), val, dtype=sim_dtype, device=dev)
    if backend.network is not None:
        ci = backend.network.node_cell[node_idx]
    else:
        ci = node_idx % backend.n_cells
    return raw[ci]


def gt_affine_for_nodes(
    params, node_idx, backend: ModelBackend, *, sim_dtype=SIM_DTYPE, session=None,
):
    """Per-node ``(a_gt, effective_bias)`` for cost / plot affine on gt.

    ``effective_bias = bias_gt``; if ``v_th`` is in ``params`` and not
    ``bias_gt_from_v_onset``, add ``v_th``. Callers must
    :func:`materialize_from_opts` so onset / alias flags are already in ``params``.
    """
    scale = _param_for_nodes(params, "a_gt", node_idx, backend, sim_dtype=sim_dtype)
    bias = _param_for_nodes(params, "bias_gt", node_idx, backend, sim_dtype=sim_dtype)
    opts = (session.train_opts if session is not None else None) or {}
    from_onset = bool(opts.get("bias_gt_from_v_onset", DEFAULT_BIAS_GT_FROM_V_ONSET))
    if (not from_onset) and "v_th" in params:
        bias = bias + _param_for_nodes(params, "v_th", node_idx, backend, sim_dtype=sim_dtype)
    return scale, bias


def _session_bias_gt_from_v_onset(session: TrainSession) -> bool:
    opts = session.train_opts or {}
    return bool(opts.get("bias_gt_from_v_onset", DEFAULT_BIAS_GT_FROM_V_ONSET))


def _session_filter(session: TrainSession) -> str:
    opts = session.train_opts or {}
    return str(opts.get("filter", DEFAULT_FILTER))


def _forward_readout_and_onset_trace(session, params, i_sti, pack):
    """``(readout_trace, onset_bias_trace)``; onset uses readout trace."""
    v = forward_v(session, params, i_sti, pack=pack)
    if _session_filter(session) != "ca":
        return v, v
    t_onset = pack_t_onset(pack)
    ca = ca_from_v_ca(v_ca_from_v(v, params, session), params, session, t_onset=t_onset)
    return ca, ca


def _pack_gt_affine_for_cost(
    params,
    pack: ReadoutPack,
    session: TrainSession,
    onset_trace: Optional[torch.Tensor] = None,
    *,
    batch_offset: int = 0,
    batch_idx=None,
):
    """Schema ``a_gt`` / ``bias_gt`` after :func:`materialize_from_opts`."""
    if _session_bias_gt_from_v_onset(session):
        if onset_trace is None:
            raise ValueError("bias_gt_from_v_onset requires onset_trace")
        materialize_from_opts(
            params, session, onset_trace=onset_trace, t_onset=pack_t_onset(pack),
        )
    _ = batch_offset
    a_gt, bias_gt = gt_affine_for_nodes(
        params, pack.readout_node, session.backend,
        sim_dtype=session.sim_dtype, session=session,
    )
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        a_gt = a_gt[mask]
        bias_gt = bias_gt[mask]
    return a_gt, bias_gt


def _session_cost_norm(session: TrainSession) -> str:
    """``cost_norm`` from train_opts; default ``a_gt2``."""
    opts = session.train_opts or {}
    raw = opts.get("cost_norm", DEFAULT_COST_NORM)
    return expand_cost_norm(raw)


def _gather_cost_time(pack: ReadoutPack, v_readout: torch.Tensor, gt: torch.Tensor):
    """Sparse ``cost_time_idx`` gather; ``cost_time_mask`` on ``v_readout`` device."""
    if pack.cost_time_idx is not None:
        idx = pack.cost_time_idx.to(device=v_readout.device)
        v_readout = v_readout.index_select(1, idx)
        gt = gt.index_select(1, idx)
    time_mask = pack.cost_time_mask
    if time_mask is not None:
        time_mask = time_mask.to(device=v_readout.device)
    return v_readout, gt, time_mask


def _weighted_mse_terms(
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    gt: torch.Tensor,
    weight: torch.Tensor,
    v_readout: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
):
    """``gt_scaled``, ``cost_weight`` (n,t), weighted SSE and gt-power (n,t)."""
    gt_scaled = a_gt[:, None] * gt
    cost_weight = weight[:, None]
    if time_mask is not None:
        cost_weight = cost_weight * time_mask.to(dtype=cost_weight.dtype, device=cost_weight.device)
    sse = cost_weight * (v_readout - gt_scaled - bias_gt[:, None]) ** 2
    power = cost_weight * gt_scaled ** 2
    return gt_scaled, cost_weight, sse, power


def _parts_from_entries(
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    gt: torch.Tensor,
    weight: torch.Tensor,
    v_readout: torch.Tensor,
    part_idx: torch.Tensor,
    keys: List[str],
    session: TrainSession,
    time_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Local costs for parts via ``part_idx`` (``part_idx`` -1 = skip entry)."""
    if not keys:
        return {}
    cost_norm = _session_cost_norm(session)
    _gt_scaled, _cost_weight, sse_wt, power_wt = _weighted_mse_terms(
        a_gt, bias_gt, gt, weight, v_readout, time_mask=time_mask,
    )
    sse_entry = sse_wt.sum(dim=-1)
    n_parts = len(keys)
    keep_f = (part_idx >= 0).to(dtype=sse_entry.dtype)
    part_idx_pos = part_idx.clamp(min=0)
    if cost_norm == "gt_power":
        p_entry = power_wt.sum(dim=-1)
        sse = sse_entry.new_zeros((n_parts,))
        gt_power = p_entry.new_zeros((n_parts,))
        sse.scatter_add_(0, part_idx_pos, sse_entry * keep_f)
        gt_power.scatter_add_(0, part_idx_pos, p_entry * keep_f)
        gt_power = torch.where(gt_power > 0, gt_power, torch.ones_like(gt_power))
        costs = sse / gt_power * 100.0
    elif cost_norm == "a_gt2":
        a2 = (a_gt * a_gt).clamp(min=torch.finfo(a_gt.dtype).tiny)
        contrib = sse_entry / a2
        costs = contrib.new_zeros((n_parts,))
        costs.scatter_add_(0, part_idx_pos, contrib * keep_f)
    else:
        raise ValueError(f"cost_norm must be one of {COST_NORMS}; got {cost_norm!r}")
    out: Dict[str, torch.Tensor] = {}
    for part_slot_idx, part_key in enumerate(keys):
        if _part_weight(session, part_key) == 0.0:
            continue
        out[part_key] = costs[part_slot_idx]
    return out


def _entries_by_part(
    pack: ReadoutPack,
    backend: ModelBackend,
    entry_key,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idx, part_keys)`` from ``entry_key(i, names, ci)``; one CPU sync."""
    n = int(pack.readout_node.shape[0])
    net = backend.network
    if net is None:
        raise ValueError("per-cell cost parts require backend.network")
    ci = net.node_cell[pack.readout_node].detach().cpu().numpy()
    entry_cost_weight = pack.cost_weight.detach().cpu().numpy()
    names = net.cells
    idx_from_part_key: Dict[str, int] = {}
    keys: List[str] = []
    part_idx_np = np.full(n, -1, dtype=np.int64)
    for entry_i in range(n):
        if entry_cost_weight[entry_i] <= 0.0:
            continue
        key = entry_key(entry_i, names, ci)
        slot_idx = idx_from_part_key.get(key)
        if slot_idx is None:
            slot_idx = len(keys)
            idx_from_part_key[key] = slot_idx
            keys.append(key)
        part_idx_np[entry_i] = slot_idx
    part_idx = torch.as_tensor(
        part_idx_np, dtype=torch.long, device=pack.readout_node.device,
    )
    return part_idx, keys


def _spot_entries_by_part(
    pack: ReadoutPack, backend: ModelBackend,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idx, part_keys)`` for spot cell×radius; one CPU sync of entry meta."""
    rad = pack.spot_cost_radius.detach().cpu().numpy()

    def entry_key(i, names, ci):
        return spot_cost_part_key(pack.name, str(names[int(ci[i])]), float(rad[i]))

    return _entries_by_part(pack, backend, entry_key)


def _moving_bar_entries_by_part(
    pack: ReadoutPack, backend: ModelBackend,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idx, part_keys)`` for moving-bar cell×PD/ND; one CPU sync of entry meta."""
    n = int(pack.readout_node.shape[0])
    if pack.cost_pd_nd is None:
        return (
            torch.full((n,), -1, dtype=torch.long, device=pack.readout_node.device),
            [],
        )
    pd_nd = pack.cost_pd_nd.detach().cpu().numpy()

    def entry_key(i, names, ci):
        lab = PD_ND_LABELS[int(pd_nd[i])]
        return moving_bar_cell_cost_part_key(pack.name, str(names[int(ci[i])]), lab)

    return _entries_by_part(pack, backend, entry_key)


def _part_weight(session: TrainSession, part_key: str) -> float:
    """Weight for a cost part; fine keys inherit coarse ``cost_weights``."""
    w = session.cost_weights or {}
    if part_key in w:
        return float(w[part_key])
    for coarse in coarse_weight_keys_for_part(part_key):
        if coarse in w:
            return float(w[coarse])
    return 1.0


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


def _mse_active_entry_mask(pack: ReadoutPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost entries with non-zero PD/ND (or pack) weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    if pack.name in MOVING_BAR_TASKS:
        if not _pack_has_active_mse(pack, session) or pack.cost_pd_nd is None:
            return torch.zeros(n, dtype=torch.bool, device=dev)
        mask = torch.zeros(n, dtype=torch.bool, device=dev)
        for pd_nd_idx, label in ((PD_IDX, "PD"), (ND_IDX, "ND")):
            if _part_weight(session, moving_bar_cost_part_key(pack.name, label)) != 0.0:
                mask |= pack.cost_pd_nd == int(pd_nd_idx)
        return mask
    on = _part_weight(session, pack.name) != 0.0
    return torch.full((n,), on, dtype=torch.bool, device=dev)


def _dsi_active_entry_mask(pack: ReadoutPack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost entries needed by a non-zero DSI weight."""
    n = int(pack.readout_batch.shape[0])
    dev = pack.readout_batch.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
    if (
        pack.dsi_pos_entries is None
        or pack.dsi_pos_entries.numel() == 0
        or _part_weight(session, dsi_key) == 0.0
    ):
        return mask
    mask[pack.dsi_pos_entries] = True
    mask[pack.dsi_neg_entries] = True
    return mask


def _pack_active_batch_indices(pack: ReadoutPack, session: TrainSession) -> Tuple[int, ...]:
    """Stimulus batch indices with at least one non-zero-weight cost node."""
    entry_mask = _mse_active_entry_mask(pack, session) | _dsi_active_entry_mask(pack, session)
    if not bool(entry_mask.any()):
        return ()
    batches = pack.readout_batch[entry_mask].unique(sorted=True)
    return tuple(int(b) for b in batches.tolist())


def _active_entry_indices(
    pack: ReadoutPack,
    session: TrainSession,
    batch_idx: Optional[int] = None,
) -> Optional[torch.Tensor]:
    keep = _mse_active_entry_mask(pack, session) | _dsi_active_entry_mask(pack, session)
    if batch_idx is not None:
        keep = keep & (pack.readout_batch == int(batch_idx))
    if not bool(keep.any()):
        return None
    return torch.nonzero(keep, as_tuple=False).reshape(-1)


def _pack_entry_fields(
    pack: ReadoutPack,
    sel,
    *,
    readout_batch: Optional[torch.Tensor] = None,
    dsi_sel=None,
) -> dict:
    """Indexed cost-entry tensors for ``replace``; ``dsi_sel`` defaults to ``sel``."""
    fields = {
        "gt": pack.gt[sel],
        "cost_weight": pack.cost_weight[sel],
        "readout_batch": (
            pack.readout_batch[sel] if readout_batch is None else readout_batch
        ),
        "readout_node": pack.readout_node[sel],
    }
    for name in ("cost_t0", "spot_cost_radius", "cost_stim_u", "cost_stim_v", "cost_pd_nd"):
        t = getattr(pack, name)
        if t is not None:
            fields[name] = t[sel]
    fields.update(remap_dsi_entries(pack, sel if dsi_sel is None else dsi_sel))
    return fields


def _slice_pack_entries(pack: ReadoutPack, entry_idx: torch.Tensor) -> ReadoutPack:
    return replace(pack, **_pack_entry_fields(pack, entry_idx))


def _subset_pack_batches(pack: ReadoutPack, batch_indices: Tuple[int, ...]) -> Optional[ReadoutPack]:
    if len(batch_indices) == int(pack.i_sti.shape[0]):
        return pack
    dev = pack.i_sti.device
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
    fields = _pack_entry_fields(pack, keep, readout_batch=new_rb, dsi_sel=kept_old)
    fields["i_sti"] = pack.i_sti.index_select(0, idx_t)
    return replace(pack, **fields)


def _pack_for_active_cost(
    pack: ReadoutPack,
    session: TrainSession,
    *,
    batch_idx: Optional[int] = None,
    batch_indices: Optional[Tuple[int, ...]] = None,
) -> Optional[ReadoutPack]:
    """Drop zero-weight entries and, when requested, inactive stimulus batches."""
    work = pack
    if batch_indices is not None:
        work = _subset_pack_batches(pack, batch_indices)
        if work is None:
            return None
    entries = _active_entry_indices(work, session, batch_idx=batch_idx)
    if entries is None:
        return None
    return _slice_pack_entries(work, entries)


def _build_cost_subpacks(session: TrainSession) -> Dict[str, ReadoutPack]:
    """Active cost entry/batch subsets per task (batched mode only)."""
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


def _i_sti_fuse_key(pack: ReadoutPack) -> Tuple:
    """Key for packs that can share one readout forward (shape, onset, polarity)."""
    i_sti = pack.i_sti
    return (
        int(i_sti.shape[1]),
        int(i_sti.shape[2]),
        str(i_sti.device),
        i_sti.dtype,
        pack_t_onset(pack),
        str(pack.name),
    )


def _build_fused_forward(
    session: TrainSession,
    cost_subpacks: Dict[str, ReadoutPack],
) -> Tuple[FusedForward, ...]:
    if session.sequential or not cost_subpacks:
        return ()
    by_key: Dict[Tuple, List[ReadoutPack]] = {}
    for pack in cost_subpacks.values():
        by_key.setdefault(_i_sti_fuse_key(pack), []).append(pack)
    fused: List[FusedForward] = []
    for packs in by_key.values():
        offsets: List[int] = []
        off = 0
        for pack in packs:
            offsets.append(off)
            off += int(pack.i_sti.shape[0])
        fused.append(FusedForward(subpacks=tuple(packs), batch_offsets=tuple(offsets)))
    return tuple(fused)


def _readout_from_trace_full(
    trace_full: torch.Tensor,
    pack: ReadoutPack,
    *,
    batch_offset: int = 0,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    rb = pack.readout_batch if batch_offset == 0 else pack.readout_batch + batch_offset
    t0 = pack_t_onset(pack)
    n_t = int(pack.gt.shape[1])
    v_readout_dsi = trace_full[rb, t0:t0 + n_t, pack.readout_node]
    if not pack_needs_waveform_mse(pack):
        return None, v_readout_dsi
    if pack.cost_t0 is None:
        return v_readout_dsi, v_readout_dsi
    v_readout = window_time_traces(
        trace_full, rb, pack.readout_node, pack.cost_t0,
        n_t, t_onset=t0,
    )
    return v_readout, v_readout_dsi


def _pack_cost_dsi_from_v_readout_dsi(
    pack: ReadoutPack,
    session: TrainSession,
    bias_gt: torch.Tensor,
    v_readout_dsi: torch.Tensor,
) -> Optional[torch.Tensor]:
    """DSI cost from full post-stimulus traces; independent of cost windows."""
    key = moving_bar_cost_part_key(pack.name, "DSI")
    if _part_weight(session, key) == 0.0:
        return None
    return cost_dsi_from_v_readout_dsi(pack, bias_gt, v_readout_dsi)


def _pack_cost_parts_from_v_readout(
    pack: ReadoutPack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: Optional[torch.Tensor],
    v_readout_dsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    backend = session.backend
    if pack.name in MOVING_BAR_TASKS:
        out: Dict[str, torch.Tensor] = {}
        if pack.cost_pd_nd is not None:
            part_idx, keys = _moving_bar_entries_by_part(pack, backend)
            if v_readout is None:
                if any(_part_weight(session, key) != 0.0 for key in keys):
                    raise ValueError(
                        f"waveform readout required for {pack.name} "
                        "PD/ND but pack has no cost_window readout",
                    )
            else:
                out.update(
                    _parts_from_entries(
                        a_gt, bias_gt, pack.gt, pack.cost_weight, v_readout,
                        part_idx, keys, session,
                    )
                )
        dsi_part = _pack_cost_dsi_from_v_readout_dsi(
            pack, session, bias_gt, v_readout_dsi,
        )
        if dsi_part is not None:
            out[moving_bar_cost_part_key(pack.name, "DSI")] = dsi_part
        return out
    if v_readout is None:
        raise ValueError(f"waveform readout required for pack {pack.name!r}")
    # #4 sparse time: gather on ``cost_time_idx``; gt_power uses a_gt·gt (no bias).
    v_readout, gt, time_mask = _gather_cost_time(pack, v_readout, pack.gt)
    if pack.spot_cost_radius is None:
        raise ValueError(f"spot pack {pack.name!r} missing spot_cost_radius")
    part_idx, keys = _spot_entries_by_part(pack, backend)
    return _parts_from_entries(
        a_gt, bias_gt, gt, pack.cost_weight, v_readout, part_idx, keys, session,
        time_mask=time_mask,
    )


def _calc_cost_parts_fused(
    params,
    session: TrainSession,
    fused_forward: Tuple[FusedForward, ...],
) -> Dict[str, torch.Tensor]:
    parts: Dict[str, torch.Tensor] = {}
    for fused in fused_forward:
        if len(fused.subpacks) == 1:
            i_sti = fused.subpacks[0].i_sti
        else:
            i_sti = torch.cat([pack.i_sti for pack in fused.subpacks], dim=0)
        # Same fuse key ⇒ shared t_onset; pass one subpack for prepare.
        trace_full, onset_trace = _forward_readout_and_onset_trace(
            session, params, i_sti, fused.subpacks[0],
        )
        for pack, off in zip(fused.subpacks, fused.batch_offsets):
            a_gt, bias_gt = _pack_gt_affine_for_cost(
                params, pack, session, onset_trace, batch_offset=off,
            )
            v_readout, v_readout_dsi = _readout_from_trace_full(
                trace_full, pack, batch_offset=off,
            )
            for key, part in _pack_cost_parts_from_v_readout(
                pack, session, a_gt, bias_gt, v_readout, v_readout_dsi,
            ).items():
                if _part_weight(session, key) != 0.0:
                    parts[key] = part
    return parts


def _pack_cost_forward(params, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        if not bool(mask.any()):
            return None
    i_sti = pack.i_sti if batch_idx is None else pack.i_sti[batch_idx:batch_idx + 1]
    trace_full, onset_trace = _forward_readout_and_onset_trace(
        session, params, i_sti, pack,
    )
    a_gt, bias_gt = _pack_gt_affine_for_cost(
        params, pack, session, onset_trace, batch_idx=batch_idx,
    )
    pd_nd = pack.cost_pd_nd
    if batch_idx is not None:
        mask = pack.readout_batch == int(batch_idx)
        gt = pack.gt[mask]
        weight = pack.cost_weight[mask]
        if pd_nd is not None:
            pd_nd = pd_nd[mask]
        rb = torch.zeros(
            int(mask.sum()), dtype=torch.long, device=pack.readout_node.device,
        )
        t0 = pack_t_onset(pack)
        n_t = int(pack.gt.shape[1])
        u_m = pack.readout_node[mask]
        v_readout_dsi = trace_full[0, t0:t0 + n_t, u_m].transpose(0, 1)
        if not pack_needs_waveform_mse(pack):
            v_readout = None
        elif pack.cost_t0 is None:
            v_readout = v_readout_dsi
        else:
            v_readout = window_time_traces(
                trace_full, rb, u_m, pack.cost_t0[mask],
                n_t, t_onset=t0,
            )
    else:
        gt = pack.gt
        weight = pack.cost_weight
        v_readout, v_readout_dsi = _readout_from_trace_full(trace_full, pack)
    return a_gt, bias_gt, gt, weight, v_readout, v_readout_dsi, pd_nd


def _pack_cost_parts_from_params(params, pack: ReadoutPack, session: TrainSession, batch_idx=None):
    """Unweighted cost parts for one pack (PD/ND split for moving_bar)."""
    fwd = _pack_cost_forward(params, pack, session, batch_idx)
    if fwd is None:
        return {}
    a_gt, bias_gt, _gt, _weight, v_readout, v_readout_dsi, _pd_nd = fwd
    return _pack_cost_parts_from_v_readout(
        pack, session, a_gt, bias_gt, v_readout, v_readout_dsi,
    )


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-part unweighted cost (before ``cost_weights``)."""
    params = params_from_z(z, session)
    cost_subpacks = _build_cost_subpacks(session)
    fused_forward = _build_fused_forward(session, cost_subpacks)
    if fused_forward:
        return _calc_cost_parts_fused(params, session, fused_forward)
    parts: Dict[str, torch.Tensor] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    if cost_subpacks and not session.sequential:
        for _name, pack in cost_subpacks.items():
            pack_parts = _pack_cost_parts_from_params(params, pack, session, batch_idx=None)
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
                        params, sub, session, batch_idx=b,
                    ).items():
                        pack_parts[key] = pack_parts.get(key, zero) + part
            dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
            if _part_weight(session, dsi_key) != 0.0:
                for batch_set in _dsi_sequential_batch_sets(pack, session):
                    sub_dsi = _pack_for_dsi_batch_set(pack, session, batch_set)
                    if sub_dsi is None:
                        continue
                    dsi_parts = _pack_cost_parts_from_params(
                        params, sub_dsi, session, batch_idx=None,
                    )
                    if dsi_key in dsi_parts:
                        pack_parts[dsi_key] = (
                            pack_parts.get(dsi_key, zero) + dsi_parts[dsi_key]
                        )
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is None:
                continue
            pack_parts = _pack_cost_parts_from_params(params, sub, session, batch_idx=None)
        for part_key, part in pack_parts.items():
            if _part_weight(session, part_key) == 0.0:
                continue
            parts[part_key] = part
    return parts


def _session_part_weight_sum(session: TrainSession) -> float:
    """Σ W over discovered fine keys (e.g. 13·(1+6·1/6)=26 for spot center+r1)."""
    keys = session_cost_part_keys(session.tasks, session=session)
    return float(sum(w for k in keys if (w := _part_weight(session, k)) != 0.0))


def _weighted_cost_from_parts(parts: Dict[str, torch.Tensor], session: TrainSession):
    """Mean of local-% parts: ``Σ W·cost / Σ W``."""
    total = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    w_sum = 0.0
    for name, part in parts.items():
        w = _part_weight(session, name)
        if w == 0.0:
            continue
        total = total + w * part
        w_sum += w
    if w_sum == 0.0:
        return total
    return total / w_sum


def calc_cost(z, session: TrainSession):
    return _weighted_cost_from_parts(calc_cost_parts(z, session), session)


def _pack_spec_names(session: TrainSession, pack: ReadoutPack) -> Tuple[str, ...]:
    opts = ((session.train_opts or {}).get(f"{pack.name}_stimulus_opts")) or {}
    names = opts.get("spec_names")
    if names:
        return tuple(str(s) for s in names)
    return tuple(s.name for s in bar_specs_for_session(session, pack.name))


def _dsi_sequential_batch_sets(
    pack: ReadoutPack, session: TrainSession,
) -> Tuple[Tuple[int, ...], ...]:
    """Active DSI microbatches: each batch_set is one axis x width (typically B=2)."""
    active = set(_pack_active_batch_indices(pack, session))
    batch_sets: list[tuple[int, ...]] = []
    for batch_set in dsi_sequential_batch_sets(_pack_spec_names(session, pack)):
        kept = tuple(b for b in batch_set if b in active)
        if len(kept) < 2:
            continue
        batch_sets.append(kept)
    return tuple(batch_sets)


def _pack_for_dsi_batch_set(
    pack: ReadoutPack,
    session: TrainSession,
    batch_set: Tuple[int, ...],
) -> Optional[ReadoutPack]:
    """Subset to one DSI batch_set; keep parent ``dsi_power`` for additive costs."""
    sub = _pack_for_active_cost(pack, session, batch_indices=batch_set)
    if sub is None or pack.dsi_power is None:
        return sub
    return replace(sub, dsi_power=pack.dsi_power)


def _iter_cost_microbatches(session: TrainSession):
    """Yield ``(pack, batch_idx, sub_pack)`` to accumulate gradients."""
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
                for batch_set in _dsi_sequential_batch_sets(pack, session):
                    sub_dsi = _pack_for_dsi_batch_set(pack, session, batch_set)
                    if sub_dsi is not None:
                        yield pack, None, sub_dsi
        else:
            sub = _pack_for_active_cost(pack, session, batch_indices=active_batches)
            if sub is not None:
                yield pack, None, sub


def backward_accumulate_weighted_cost(z, session: TrainSession):
    """Backward mean local-% cost one micro-batch at a time.

    Each microbatch contributes ``Σ (W·cost) / W_norm`` so gradients match
    ``calc_cost`` (``Σ W·cost / Σ W``).
    """
    part_sums: Dict[str, float] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    w_norm = _session_part_weight_sum(session)
    if w_norm == 0.0:
        return 0.0, {}
    for pack, batch_idx, sub in _iter_cost_microbatches(session):
        params = params_from_z(z, session)
        mb_loss = zero
        has_loss = False
        dsi_only = (
            session.sequential
            and batch_idx is None
            and pack.name in MOVING_BAR_TASKS
        )
        dsi_key = moving_bar_cost_part_key(pack.name, "DSI")
        for key, part in _pack_cost_parts_from_params(
            params, sub, session, batch_idx=batch_idx,
        ).items():
            if dsi_only and key != dsi_key:
                continue
            if (not dsi_only) and session.sequential and key == dsi_key:
                # single-batch slices have no complete DSI groups; skip zeros
                continue
            w = _part_weight(session, key)
            if w == 0.0:
                continue
            mb_loss = mb_loss + (w / w_norm) * part
            has_loss = True
            part_sums[key] = part_sums.get(key, 0.0) + float(part.item())
        if has_loss:
            mb_loss.backward()
    total = sum(_part_weight(session, k) * v for k, v in part_sums.items()) / w_norm
    return total, part_sums
