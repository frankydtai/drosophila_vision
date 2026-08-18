"""Cost assembly: per-part MSE / DSI and the scaled total.

Consumes a :class:`~train.session.TrainSession` and the model forward
(``neuron.forward`` + ``neuron.readout``); produces per-part local-% costs and
the mean scaled total (``Σ W·cost / Σ W``). The staged-lr loop lives in
:mod:`train.optimization`.

Owns cost-time execution plans (active packs / fused packs) built at
calc time — not at session open.

Readout traces are absolute ``v`` (``filter=none``) or ``ca`` (``filter=ca``);
cost compares the readout to ``gt_aff = a_gt * gt + bias_gt`` (``+ v_th`` when
present and ``val_from`` bias_gt is off). When ``train_opts['val_from']['bias_gt']`` is
enabled with source ``v_onset``, ``params['bias_gt']`` is written from ``v`` at ``t_onset``
(or ``v_ca`` when ``filter=ca``), clamped to schema ``bias_gt`` ``lo``/``hi`` (always in graph).
Same for ``v_th_ca`` / ``a_ca`` ``val_from`` entries:
sources are written into ``params['v_th_ca']`` / ``params['a_ca']``
(and ``param.csv``).
Waveform MSE normalization is ``session`` / ``train_opts`` ``cost_norm``:

* ``gt_power``: ``100 * Σ W (v_readout−gt_aff)² / Σ W (a_gt·gt)²``
  (no ``bias_gt`` / ``v_th`` in the denominator)
* ``a_gt2`` (default): ``Σ W (v_readout−gt_aff)² / a_gt²`` (per-entry ``a_i²``; bias not in denom)

**within each part**. The train total averages those part costs (equal
scale per cell×radius unless ``part_cost_scales`` says otherwise).

Sparse cost time points (#4): ``pack.gts`` stays the ``ms_response`` window length
(spot excludes ``ms_post``) and the subsample is gathered from both v_readout
trace and gt at cost time via ``pack.cost_ts`` (spread: ``cost_interval_ms``;
spot: ``cost_interval_ms`` / ``cost_ms``); when radii use different
``cost_ms`` lists, ``pack.cost_time_mask`` zeros non-participating (entry, t)
pairs. ``gt_power`` is recomputed on the subsample.
"""
from __future__ import annotations

from config import (
    TRAIN_OPTIMIZATION,
)

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neuron.forward import (
    forward_trace,
    pack_t_onset,
)
from neuron.readout import (
    pack_needs_waveform_mse,
    window_time_traces,
)

from train.param import (
    SIM_DTYPE,
    override_val_from,
    params_from_z,
    val_from_enabled,
)
from train.session import Pack, TrainSession
from task.moving_bar.gt import dsi_sequential_b_sets
from task.moving_bar.pack import (
    bar_specs_from_task,
    cost_dsi_from_v_readout_dsi,
    remap_dsi_entries,
)
from task.moving_bar.sti_spec import PD_ND_LABELS


COST_NORMS = ("gt_power", "a_gt2")


def spread_cost_part_key(task: str, contrast: str, cell: str) -> str:
    return f"{task}_{contrast}_{cell}"


def spot_cost_part_key(task: str, contrast: str, cell: str, radius) -> str:
    return f"{task}_{contrast}_{cell}_r{int(radius)}"


def moving_bar_cell_cost_part_key(task: str, contrast: str, cell: str, part: str) -> str:
    return f"{task}_{contrast}_{cell}_{part}"


def moving_bar_cost_part_key(task: str, contrast: str, part: str) -> str:
    return f"{task}_{contrast}_{part}"


@dataclass(frozen=True)
class FusedPacks:
    """Packs with matching i_sti shape / onset; one ``forward`` per fused group."""

    packs: Tuple[Pack, ...]
    b_offsets: Tuple[int, ...]


def pack_cost_abs_ts(pack: Pack, t_onset, *, entry_radius=None):
    """Absolute cost ``ts`` for sparse spot cost samples (or ``None``).

    Sole reader of ``cost_ts`` / ``cost_time_mask`` / ``entry_radii``.
    ``entry_radius`` is one hex-lattice radius; when set and a mask exists, keep
    that radius's columns only. Omit ``entry_radius`` → union of all radii.
    """
    cost_ts = pack.cost_ts
    if cost_ts is None:
        return None
    base = int(t_onset or 0)
    ts = cost_ts.detach().cpu().numpy().astype(np.int64, copy=False)
    if entry_radius is None:
        return base + ts
    mask = pack.cost_time_mask
    entry_radii = pack.entry_radii
    if mask is None or entry_radii is None:
        return base + ts
    entry_radii = entry_radii.detach().cpu().numpy().astype(np.int64, copy=False)
    hit = np.where(entry_radii == int(entry_radius))[0]
    if not hit.size:
        return base + np.zeros(0, dtype=np.int64)
    entry_mask = mask[int(hit[0])].detach().cpu().numpy() > 0
    return base + ts[entry_mask]


def node_vals_from_param(params, param: str, nodes, connectome, *, sim_dtype=SIM_DTYPE):
    """Per-node vals from a cell-indexed schema param (or scalar default)."""
    vals = params.get(param, 1.0 if param == "a_gt" else 0.0)
    n = int(nodes.shape[0])
    device = nodes.device
    if not torch.is_tensor(vals) or vals.dim() == 0:
        val = float(vals if not torch.is_tensor(vals) else vals.item())
        return torch.full((n,), val, dtype=sim_dtype, device=device)
    ci = connectome.node_cells[nodes]
    return vals[ci]


def gt_affine_from_nodes(
    params, nodes, connectome, *, sim_dtype=SIM_DTYPE, session=None,
):
    """Per-node ``(a_gt, effective_bias)`` for cost / plot affine on gt.

    ``effective_bias = bias_gt``; if ``v_th`` is in ``params`` and not
    ``val_from`` bias_gt is off, add ``v_th``. Callers must
    :func:`override_val_from` so ``val_from`` sources are already in ``params``.
    """
    a_gt = node_vals_from_param(params, "a_gt", nodes, connectome, sim_dtype=sim_dtype)
    bias = node_vals_from_param(params, "bias_gt", nodes, connectome, sim_dtype=sim_dtype)
    opts = (session.train_opts if session is not None else None) or {}
    from_onset = val_from_enabled(opts, "bias_gt")
    if (not from_onset) and "v_th" in params:
        bias = bias + node_vals_from_param(params, "v_th", nodes, connectome, sim_dtype=sim_dtype)
    return a_gt, bias


def gt_affine_from_cell(
    params, cell, connectome, *, sim_dtype=SIM_DTYPE, session=None,
) -> tuple[float, float]:
    """Cell-level ``(a_gt, effective_bias)`` — mean over that cell's nodes."""
    cell_idx = [str(name) for name in connectome.cells].index(str(cell))
    nodes = torch.nonzero(
        connectome.node_cells == cell_idx, as_tuple=False,
    ).squeeze(-1)
    a_gt, bias = gt_affine_from_nodes(
        params, nodes, connectome, sim_dtype=sim_dtype, session=session,
    )
    return float(a_gt[0].item()), float(bias.mean().item())


def forward_pack(session, params, i_sti, pack):
    """Readout trace ``(B, T, N)`` then ``val_from`` from onset (mutates ``params``)."""
    trace = forward_trace(session, params, i_sti, pack=pack)
    override_val_from(params, session, onset_trace=trace, t_onset=pack_t_onset(pack))
    return trace


def gt_affine_from_pack(
    params,
    pack: Pack,
    session: TrainSession,
    *,
    b=None,
):
    """Schema ``a_gt`` / ``bias_gt`` after :func:`forward_pack` ``val_from``."""
    a_gt, bias_gt = gt_affine_from_nodes(
        params, pack.entry_nodes, session.connectome,
        sim_dtype=session.sim_dtype, session=session,
    )
    if b is not None:
        mask = pack.entry_bs == int(b)
        a_gt = a_gt[mask]
        bias_gt = bias_gt[mask]
    return a_gt, bias_gt


def _session_cost_norm(session: TrainSession) -> str:
    """``cost_norm`` from train_opts; default ``a_gt2``."""
    opts = session.train_opts or {}
    return str(opts.get("cost_norm", TRAIN_OPTIMIZATION['cost_norm']))


def _gather_cost_time(pack: Pack, v_readout: torch.Tensor, gts: torch.Tensor):
    """Sparse ``cost_ts`` gather; ``cost_time_mask`` on ``v_readout`` device."""
    if pack.cost_ts is not None:
        cost_ts = pack.cost_ts.to(device=v_readout.device)
        v_readout = v_readout.index_select(1, cost_ts)
        gts = gts.index_select(1, cost_ts)
    time_mask = pack.cost_time_mask
    if time_mask is not None:
        time_mask = time_mask.to(device=v_readout.device)
    return v_readout, gts, time_mask


def _scaled_mse_terms(
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    gts: torch.Tensor,
    scale: torch.Tensor,
    v_readout: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
):
    """``gt_scaled``, ``cost_scales`` (n,t), scaled SSE and gt-power (n,t)."""
    gt_scaled = a_gt[:, None] * gts
    cost_scales = scale[:, None]
    if time_mask is not None:
        cost_scales = cost_scales * time_mask.to(dtype=cost_scales.dtype, device=cost_scales.device)
    sse = cost_scales * (v_readout - gt_scaled - bias_gt[:, None]) ** 2
    power = cost_scales * gt_scaled ** 2
    return gt_scaled, cost_scales, sse, power


def _parts_from_entries(
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    gts: torch.Tensor,
    scale: torch.Tensor,
    v_readout: torch.Tensor,
    part_idxs: torch.Tensor,
    part_keys: List[str],
    session: TrainSession,
    time_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Local costs for parts via ``part_idxs`` (``part_idxs`` -1 = skip entry)."""
    if not part_keys:
        return {}
    cost_norm = _session_cost_norm(session)
    _gt_scaled, _cost_scale, sse_wt, power_wt = _scaled_mse_terms(
        a_gt, bias_gt, gts, scale, v_readout, time_mask=time_mask,
    )
    sse_entry = sse_wt.sum(dim=-1)
    n_part = len(part_keys)
    keep_f = (part_idxs >= 0).to(dtype=sse_entry.dtype)
    part_idxs_pos = part_idxs.clamp(min=0)
    if cost_norm == "gt_power":
        p_entry = power_wt.sum(dim=-1)
        sse = sse_entry.new_zeros((n_part,))
        gt_power = p_entry.new_zeros((n_part,))
        sse.scatter_add_(0, part_idxs_pos, sse_entry * keep_f)
        gt_power.scatter_add_(0, part_idxs_pos, p_entry * keep_f)
        gt_power = torch.where(gt_power > 0, gt_power, torch.ones_like(gt_power))
        costs = sse / gt_power * 100.0
    elif cost_norm == "a_gt2":
        a2 = (a_gt * a_gt).clamp(min=torch.finfo(a_gt.dtype).tiny)
        contrib = sse_entry / a2
        costs = contrib.new_zeros((n_part,))
        costs.scatter_add_(0, part_idxs_pos, contrib * keep_f)
    else:
        raise ValueError(f"cost_norm must be one of {COST_NORMS}; got {cost_norm!r}")
    parts: Dict[str, torch.Tensor] = {}
    for part_idx, part_key in enumerate(part_keys):
        if _part_scale(session, part_key) == 0.0:
            continue
        parts[part_key] = costs[part_idx]
    return parts


def _entries_by_part(
    pack: Pack,
    connectome,
    entry_part_key,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idxs, part_keys)`` from ``entry_part_key(entry, cells, ci)``; one CPU sync."""
    n = int(pack.entry_nodes.shape[0])
    ci = connectome.node_cells[pack.entry_nodes].detach().cpu().numpy()
    entry_cost_scales = pack.cost_scales.detach().cpu().numpy()
    cells = connectome.cells
    part_idxs: Dict[str, int] = {}
    part_keys: List[str] = []
    part_idxs_np = np.full(n, -1, dtype=np.int64)
    for entry in range(n):
        if entry_cost_scales[entry] <= 0.0:
            continue
        part_key = entry_part_key(entry, cells, ci)
        part_idx = part_idxs.get(part_key)
        if part_idx is None:
            part_idx = len(part_keys)
            part_idxs[part_key] = part_idx
            part_keys.append(part_key)
        part_idxs_np[entry] = part_idx
    part_idxs = torch.as_tensor(
        part_idxs_np, dtype=torch.long, device=pack.entry_nodes.device,
    )
    return part_idxs, part_keys


def _spread_entries_by_part(
    pack: Pack, connectome,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idxs, part_keys)`` for spread cell; one CPU sync of entry meta."""

    def entry_part_key(entry, cells, ci):
        return spread_cost_part_key(
            pack.task, pack.contrast, str(cells[int(ci[entry])]),
        )

    return _entries_by_part(pack, connectome, entry_part_key)


def _spot_entries_by_part(
    pack: Pack, connectome,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idxs, part_keys)`` for spot cell×radius; one CPU sync of entry meta."""
    entry_radii = pack.entry_radii.detach().cpu().numpy()

    def entry_part_key(entry, cells, ci):
        return spot_cost_part_key(
            pack.task, pack.contrast, str(cells[int(ci[entry])]), int(entry_radii[entry]),
        )

    return _entries_by_part(pack, connectome, entry_part_key)


def _moving_bar_entries_by_part(
    pack: Pack, connectome,
) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idxs, part_keys)`` for moving-bar cell×PD/ND; one CPU sync of entry meta."""
    n = int(pack.entry_nodes.shape[0])
    if pack.cost_pd_nds is None:
        return (
            torch.full((n,), -1, dtype=torch.long, device=pack.entry_nodes.device),
            [],
        )
    pd_nd = pack.cost_pd_nds.detach().cpu().numpy()

    def entry_part_key(entry, cells, ci):
        part = PD_ND_LABELS[int(pd_nd[entry])]
        return moving_bar_cell_cost_part_key(
            pack.task, pack.contrast, str(cells[int(ci[entry])]), part,
        )

    return _entries_by_part(pack, connectome, entry_part_key)


def _part_scale(session: TrainSession, part_key: str) -> float:
    """Exact ``part_cost_scales[part_key]``; missing key is 1.0."""
    return float((session.part_cost_scales or {}).get(part_key, 1.0))


def _mse_entries_by_part(
    pack: Pack, session: TrainSession,
) -> Tuple[torch.Tensor, List[str]]:
    if pack.task == "moving_bar":
        return _moving_bar_entries_by_part(pack, session.connectome)
    if pack.task == "spot":
        return _spot_entries_by_part(pack, session.connectome)
    if pack.task == "spread":
        return _spread_entries_by_part(pack, session.connectome)
    raise KeyError(pack.task)


def _pack_part_keys(pack: Pack, session: TrainSession) -> List[str]:
    _, part_keys = _mse_entries_by_part(pack, session)
    if (
        pack.task == "moving_bar"
        and pack.dsi_pos_ptr is not None
        and int(pack.dsi_pos_ptr.numel()) > 1
    ):
        dsi = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
        if dsi not in part_keys:
            part_keys = [*part_keys, dsi]
    return part_keys


def session_cost_part_keys(session: TrainSession) -> Tuple[str, ...]:
    """Fine cost part_keys from session packs."""
    part_keys: List[str] = []
    for pack in session.iter_packs():
        part_keys.extend(_pack_part_keys(pack, session))
    return tuple(part_keys)


def _pack_has_active_cost(pack: Pack, session: TrainSession) -> bool:
    return any(
        _part_scale(session, part_key) != 0.0
        for part_key in _pack_part_keys(pack, session)
    )


def _pack_has_active_mse(pack: Pack, session: TrainSession) -> bool:
    _, part_keys = _mse_entries_by_part(pack, session)
    return any(_part_scale(session, part_key) != 0.0 for part_key in part_keys)


def _mse_entry_mask(pack: Pack, session: TrainSession) -> torch.Tensor:
    n = int(pack.entry_bs.shape[0])
    device = pack.entry_bs.device
    part_idxs, part_keys = _mse_entries_by_part(pack, session)
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for part_idx, part_key in enumerate(part_keys):
        if _part_scale(session, part_key) != 0.0:
            mask |= part_idxs == part_idx
    return mask


def _dsi_entry_mask(pack: Pack, session: TrainSession) -> torch.Tensor:
    """Boolean mask over cost entries needed by a non-zero DSI scale."""
    n = int(pack.entry_bs.shape[0])
    device = pack.entry_bs.device
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    dsi_part_key = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
    if (
        pack.dsi_pos_entries is None
        or pack.dsi_pos_entries.numel() == 0
        or _part_scale(session, dsi_part_key) == 0.0
    ):
        return mask
    mask[pack.dsi_pos_entries] = True
    if pack.dsi_neg_entries is not None:
        mask[pack.dsi_neg_entries] = True
    return mask


def _pack_active_bs(pack: Pack, session: TrainSession) -> Tuple[int, ...]:
    """Stimulus b idxs with at least one non-zero-scale cost node."""
    entry_mask = _mse_entry_mask(pack, session) | _dsi_entry_mask(pack, session)
    if not bool(entry_mask.any()):
        return ()
    bs = pack.entry_bs[entry_mask].unique(sorted=True)
    return tuple(int(b) for b in bs.tolist())


def _active_entries(
    pack: Pack,
    session: TrainSession,
    b: Optional[int] = None,
) -> Optional[torch.Tensor]:
    entry_mask = _mse_entry_mask(pack, session) | _dsi_entry_mask(pack, session)
    if b is not None:
        entry_mask = entry_mask & (pack.entry_bs == int(b))
    if not bool(entry_mask.any()):
        return None
    return torch.nonzero(entry_mask, as_tuple=False).reshape(-1)


def _pack_entry_fields(
    pack: Pack,
    sel,
    *,
    entry_bs: Optional[torch.Tensor] = None,
    dsi_sel=None,
) -> dict:
    """Indexed cost-entry tensors for ``replace``; ``dsi_sel`` defaults to ``sel``."""
    fields = {
        "gts": pack.gts[sel],
        "cost_scales": pack.cost_scales[sel],
        "entry_bs": (
            pack.entry_bs[sel] if entry_bs is None else entry_bs
        ),
        "entry_nodes": pack.entry_nodes[sel],
    }
    for field in ("cost_t0s", "entry_radii", "cost_sti_us", "cost_sti_vs", "cost_pd_nds"):
        t = getattr(pack, field)
        if t is not None:
            fields[field] = t[sel]
    fields.update(remap_dsi_entries(pack, sel if dsi_sel is None else dsi_sel))
    return fields


def _slice_pack_entries(pack: Pack, entries: torch.Tensor) -> Pack:
    return replace(pack, **_pack_entry_fields(pack, entries))


def _subset_pack_bs(pack: Pack, bs: Tuple[int, ...]) -> Optional[Pack]:
    if len(bs) == int(pack.i_sti.shape[0]):
        return pack
    device = pack.i_sti.device
    idx_t = torch.tensor(bs, dtype=torch.long, device=device)
    rb = pack.entry_bs
    keep = torch.isin(rb, idx_t)
    if not bool(keep.any()):
        return None
    lut_size = int(max(max(bs), int(rb.max()))) + 1
    lut = torch.full((lut_size,), -1, dtype=torch.long, device=device)
    lut[idx_t] = torch.arange(len(bs), dtype=torch.long, device=device)
    new_rb = lut[rb[keep]]
    kept_old = torch.nonzero(keep, as_tuple=False).reshape(-1)
    fields = _pack_entry_fields(pack, keep, entry_bs=new_rb, dsi_sel=kept_old)
    fields["i_sti"] = pack.i_sti.index_select(0, idx_t)
    return replace(pack, **fields)


def _active_cost_pack(
    pack: Pack,
    session: TrainSession,
    *,
    b: Optional[int] = None,
    bs: Optional[Tuple[int, ...]] = None,
) -> Optional[Pack]:
    """Drop zero-scale entries and, when requested, inactive sti bs."""
    work = pack
    if bs is not None:
        work = _subset_pack_bs(pack, bs)
        if work is None:
            return None
    entries = _active_entries(work, session, b=b)
    if entries is None:
        return None
    return _slice_pack_entries(work, entries)


def _build_active_packs(session: TrainSession) -> Tuple[Pack, ...]:
    """Active cost entry/b packs (non-sequential mode only)."""
    if session.sequential:
        return ()
    active_packs: List[Pack] = []
    for pack in session.iter_packs():
        if not _pack_has_active_cost(pack, session):
            continue
        active_bs = _pack_active_bs(pack, session)
        if not active_bs:
            continue
        pack = _active_cost_pack(pack, session, bs=active_bs)
        if pack is not None:
            active_packs.append(pack)
    return tuple(active_packs)


def _build_fused_packs(
    session: TrainSession,
    active_packs: Tuple[Pack, ...],
) -> Tuple[FusedPacks, ...]:
    """Group compatible packs for one forward each (cat ``i_sti`` along ``b``)."""
    if session.sequential or not active_packs:
        return ()
    grouped: Dict[Tuple, List[Pack]] = {}
    for pack in active_packs:
        i_sti = pack.i_sti
        grouped.setdefault(
            (
                int(i_sti.shape[1]),
                int(i_sti.shape[2]),
                str(i_sti.device),
                i_sti.dtype,
                pack_t_onset(pack),
                str(pack.task),
            ),
            [],
        ).append(pack)
    fused_packs: List[FusedPacks] = []
    for packs in grouped.values():
        offsets: List[int] = []
        off = 0
        for pack in packs:
            offsets.append(off)
            off += int(pack.i_sti.shape[0])
        fused_packs.append(FusedPacks(packs=tuple(packs), b_offsets=tuple(offsets)))
    return tuple(fused_packs)


def _readout_from_trace(
    trace: torch.Tensor,
    pack: Pack,
    *,
    b_offset: int = 0,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    rb = pack.entry_bs if b_offset == 0 else pack.entry_bs + b_offset
    t0 = pack_t_onset(pack)
    n_t = int(pack.gts.shape[1])
    v_readout_dsi = trace[rb, t0:t0 + n_t, pack.entry_nodes]
    if not pack_needs_waveform_mse(pack):
        return None, v_readout_dsi
    if pack.cost_t0s is None:
        return v_readout_dsi, v_readout_dsi
    v_readout = window_time_traces(
        trace, rb, pack.entry_nodes, pack.cost_t0s,
        n_t, t_onset=t0,
    )
    return v_readout, v_readout_dsi


def _pack_cost_dsi_from_v_readout_dsi(
    pack: Pack,
    session: TrainSession,
    bias_gt: torch.Tensor,
    v_readout_dsi: torch.Tensor,
) -> Optional[torch.Tensor]:
    """DSI cost from post-sti traces; independent of cost windows."""
    dsi_part_key = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
    if _part_scale(session, dsi_part_key) == 0.0:
        return None
    return cost_dsi_from_v_readout_dsi(pack, bias_gt, v_readout_dsi)


def _spread_pack_cost_parts_from_v_readout(
    pack: Pack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: Optional[torch.Tensor],
    v_readout_dsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if v_readout is None:
        raise ValueError(f"waveform readout required for pack {pack.task!r}")
    v_readout, gts, time_mask = _gather_cost_time(pack, v_readout, pack.gts)
    part_idxs, part_keys = _spread_entries_by_part(pack, session.connectome)
    return _parts_from_entries(
        a_gt, bias_gt, gts, pack.cost_scales, v_readout, part_idxs, part_keys, session,
        time_mask=time_mask,
    )


def _spot_pack_cost_parts_from_v_readout(
    pack: Pack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: Optional[torch.Tensor],
    v_readout_dsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if v_readout is None:
        raise ValueError(f"waveform readout required for pack {pack.task!r}")
    v_readout, gts, time_mask = _gather_cost_time(pack, v_readout, pack.gts)
    if pack.entry_radii is None:
        raise ValueError(f"spot pack {pack.task!r} missing entry_radii")
    part_idxs, part_keys = _spot_entries_by_part(pack, session.connectome)
    return _parts_from_entries(
        a_gt, bias_gt, gts, pack.cost_scales, v_readout, part_idxs, part_keys, session,
        time_mask=time_mask,
    )


def _moving_bar_pack_cost_parts_from_v_readout(
    pack: Pack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: Optional[torch.Tensor],
    v_readout_dsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    connectome = session.connectome
    parts: Dict[str, torch.Tensor] = {}
    if pack.cost_pd_nds is not None:
        part_idxs, part_keys = _moving_bar_entries_by_part(pack, connectome)
        if v_readout is None:
            if any(_part_scale(session, part_key) != 0.0 for part_key in part_keys):
                raise ValueError(
                    f"waveform readout required for {pack.task} "
                    "PD/ND but pack has no cost_window readout",
                )
        else:
            parts.update(
                _parts_from_entries(
                    a_gt, bias_gt, pack.gts, pack.cost_scales, v_readout,
                    part_idxs, part_keys, session,
                )
            )
    dsi_part = _pack_cost_dsi_from_v_readout_dsi(
        pack, session, bias_gt, v_readout_dsi,
    )
    if dsi_part is not None:
        parts[moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")] = dsi_part
    return parts


_PACK_COST_PARTS_FROM_V_READOUT = {
    "spread": _spread_pack_cost_parts_from_v_readout,
    "spot": _spot_pack_cost_parts_from_v_readout,
    "moving_bar": _moving_bar_pack_cost_parts_from_v_readout,
}


def _pack_cost_parts_from_v_readout(
    pack: Pack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: Optional[torch.Tensor],
    v_readout_dsi: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return _PACK_COST_PARTS_FROM_V_READOUT[pack.task](
        pack, session, a_gt, bias_gt, v_readout, v_readout_dsi,
    )


def _pack_cost_forward(params, pack: Pack, session: TrainSession, b=None):
    if b is not None:
        mask = pack.entry_bs == int(b)
        if not bool(mask.any()):
            return None
    i_sti = pack.i_sti if b is None else pack.i_sti[b:b + 1]
    trace = forward_pack(session, params, i_sti, pack)
    a_gt, bias_gt = gt_affine_from_pack(
        params, pack, session, b=b,
    )
    pd_nd = pack.cost_pd_nds
    if b is not None:
        mask = pack.entry_bs == int(b)
        gts = pack.gts[mask]
        scale = pack.cost_scales[mask]
        if pd_nd is not None:
            pd_nd = pd_nd[mask]
        rb = torch.zeros(
            int(mask.sum()), dtype=torch.long, device=pack.entry_nodes.device,
        )
        t0 = pack_t_onset(pack)
        n_t = int(pack.gts.shape[1])
        u_m = pack.entry_nodes[mask]
        v_readout_dsi = trace[0, t0:t0 + n_t, u_m].transpose(0, 1)
        if not pack_needs_waveform_mse(pack):
            v_readout = None
        elif pack.cost_t0s is None:
            v_readout = v_readout_dsi
        else:
            v_readout = window_time_traces(
                trace, rb, u_m, pack.cost_t0s[mask],
                n_t, t_onset=t0,
            )
    else:
        gts = pack.gts
        scale = pack.cost_scales
        v_readout, v_readout_dsi = _readout_from_trace(trace, pack)
    return a_gt, bias_gt, gts, scale, v_readout, v_readout_dsi, pd_nd


def _pack_cost_parts_from_params(params, pack: Pack, session: TrainSession, b=None):
    """Unscaled cost parts for one pack (PD/ND split for moving_bar)."""
    fwd = _pack_cost_forward(params, pack, session, b)
    if fwd is None:
        return {}
    a_gt, bias_gt, _gt, _scale, v_readout, v_readout_dsi, _pd_nd = fwd
    return _pack_cost_parts_from_v_readout(
        pack, session, a_gt, bias_gt, v_readout, v_readout_dsi,
    )


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-part unscaled cost (before ``part_cost_scales``)."""
    params = params_from_z(z, session)
    active_packs = _build_active_packs(session)
    fused_packs = _build_fused_packs(session, active_packs)
    if fused_packs:
        parts: Dict[str, torch.Tensor] = {}
        for fused in fused_packs:
            if len(fused.packs) == 1:
                i_sti = fused.packs[0].i_sti
            else:
                i_sti = torch.cat([pack.i_sti for pack in fused.packs], dim=0)
            # Compatible packs share t_onset; pass one pack for prepare.
            trace = forward_pack(
                session, params, i_sti, fused.packs[0],
            )
            for pack, off in zip(fused.packs, fused.b_offsets):
                a_gt, bias_gt = gt_affine_from_pack(
                    params, pack, session,
                )
                v_readout, v_readout_dsi = _readout_from_trace(
                    trace, pack, b_offset=off,
                )
                for part_key, part in _pack_cost_parts_from_v_readout(
                    pack, session, a_gt, bias_gt, v_readout, v_readout_dsi,
                ).items():
                    if _part_scale(session, part_key) != 0.0:
                        parts[part_key] = part
        return parts
    parts: Dict[str, torch.Tensor] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    if active_packs and not session.sequential:
        for pack in active_packs:
            pack_parts = _pack_cost_parts_from_params(params, pack, session, b=None)
            for part_key, part in pack_parts.items():
                if _part_scale(session, part_key) == 0.0:
                    continue
                parts[part_key] = part
        return parts
    for pack in session.iter_packs():
        if not _pack_has_active_cost(pack, session):
            continue
        active_bs = _pack_active_bs(pack, session)
        if not active_bs:
            continue
        if session.sequential:
            pack_parts: Dict[str, torch.Tensor] = {}
            if _pack_has_active_mse(pack, session):
                for b in active_bs:
                    active_pack = _active_cost_pack(pack, session, b=b)
                    if active_pack is None:
                        continue
                    for part_key, part in _pack_cost_parts_from_params(
                        params, active_pack, session, b=b,
                    ).items():
                        pack_parts[part_key] = pack_parts.get(part_key, zero) + part
            dsi_part_key = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
            if _part_scale(session, dsi_part_key) != 0.0:
                for b_set in _dsi_sequential_b_sets(pack, session):
                    active_pack = _dsi_b_set_pack(pack, session, b_set)
                    if active_pack is None:
                        continue
                    dsi_parts = _pack_cost_parts_from_params(
                        params, active_pack, session, b=None,
                    )
                    if dsi_part_key in dsi_parts:
                        pack_parts[dsi_part_key] = (
                            pack_parts.get(dsi_part_key, zero) + dsi_parts[dsi_part_key]
                        )
        else:
            active_pack = _active_cost_pack(pack, session, bs=active_bs)
            if active_pack is None:
                continue
            pack_parts = _pack_cost_parts_from_params(params, active_pack, session, b=None)
        for part_key, part in pack_parts.items():
            if _part_scale(session, part_key) == 0.0:
                continue
            parts[part_key] = part
    return parts


def _session_part_scale_sum(session: TrainSession) -> float:
    """Σ W over discovered fine part_keys (e.g. 13·(1+6·1/6)=26 for spot center+r1)."""
    part_keys = session_cost_part_keys(session)
    return float(sum(scale for k in part_keys if (scale := _part_scale(session, k)) != 0.0))


def _scaled_cost_from_parts(parts: Dict[str, torch.Tensor], session: TrainSession):
    """Mean of local-% parts: ``Σ W·cost / Σ W``."""
    total = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    scale_sum = 0.0
    for part_key, part in parts.items():
        scale = _part_scale(session, part_key)
        if scale == 0.0:
            continue
        total = total + scale * part
        scale_sum += scale
    if scale_sum == 0.0:
        return total
    return total / scale_sum


def calc_cost(z, session: TrainSession, parts=None):
    if parts is None:
        parts = calc_cost_parts(z, session)
    return _scaled_cost_from_parts(parts, session)


def _pack_spec_tokens(session: TrainSession, pack: Pack) -> Tuple[str, ...]:
    opts = ((session.train_opts or {}).get(f"{pack.task}_sti_opts")) or {}
    spec_tokens = opts.get("spec_tokens")
    if spec_tokens:
        return tuple(str(token) for token in spec_tokens)
    return tuple(spec.token for spec in bar_specs_from_task(session, pack.task))


def _dsi_sequential_b_sets(
    pack: Pack, session: TrainSession,
) -> Tuple[Tuple[int, ...], ...]:
    """Active DSI b_sets: each b_set is one axis x w (typically B=2)."""
    active = set(_pack_active_bs(pack, session))
    b_sets: list[tuple[int, ...]] = []
    for b_set in dsi_sequential_b_sets(_pack_spec_tokens(session, pack)):
        kept = tuple(b for b in b_set if b in active)
        if len(kept) < 2:
            continue
        b_sets.append(kept)
    return tuple(b_sets)


def _dsi_b_set_pack(
    pack: Pack,
    session: TrainSession,
    b_set: Tuple[int, ...],
) -> Optional[Pack]:
    """Pack for one DSI ``b_set``; keep parent ``dsi_power`` for additive costs."""
    dsi_power = pack.dsi_power
    pack = _active_cost_pack(pack, session, bs=b_set)
    if pack is None or dsi_power is None:
        return pack
    return replace(pack, dsi_power=dsi_power)


def _iter_cost_bs(session: TrainSession):
    """Yield ``(pack, b)`` for gradient part sums."""
    for pack in session.iter_packs():
        if not _pack_has_active_cost(pack, session):
            continue
        active_bs = _pack_active_bs(pack, session)
        if not active_bs:
            continue
        if session.sequential:
            if _pack_has_active_mse(pack, session):
                for b in active_bs:
                    active_pack = _active_cost_pack(pack, session, b=b)
                    if active_pack is not None:
                        yield active_pack, b
            dsi_part_key = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
            if _part_scale(session, dsi_part_key) != 0.0:
                for b_set in _dsi_sequential_b_sets(pack, session):
                    active_pack = _dsi_b_set_pack(pack, session, b_set)
                    if active_pack is not None:
                        yield active_pack, None
        else:
            active_pack = _active_cost_pack(pack, session, bs=active_bs)
            if active_pack is not None:
                yield active_pack, None


def backward_part_sums(z, session: TrainSession):
    """Backward mean local-% cost one ``b`` at a time.

    Each ``b`` contributes ``Σ (W·cost) / W_norm`` so gradients match
    ``calc_cost`` (``Σ W·cost / Σ W``). Returns ``(total, part_sums)``.
    """
    part_sums: Dict[str, float] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    scale_norm = _session_part_scale_sum(session)
    if scale_norm == 0.0:
        return 0.0, {}
    for pack, b in _iter_cost_bs(session):
        params = params_from_z(z, session)
        mb_loss = zero
        has_loss = False
        dsi_only = session.sequential and b is None
        dsi_part_key = moving_bar_cost_part_key(pack.task, pack.contrast, "DSI")
        for part_key, part in _pack_cost_parts_from_params(
            params, pack, session, b=b,
        ).items():
            if dsi_only and part_key != dsi_part_key:
                continue
            if (not dsi_only) and session.sequential and part_key == dsi_part_key:
                # single-b slices have no complete DSI groups; skip zeros
                continue
            scale = _part_scale(session, part_key)
            if scale == 0.0:
                continue
            mb_loss = mb_loss + (scale / scale_norm) * part
            has_loss = True
            part_sums[part_key] = part_sums.get(part_key, 0.0) + float(part.item())
        if has_loss:
            mb_loss.backward()
    total = sum(_part_scale(session, k) * v for k, v in part_sums.items()) / scale_norm
    return total, part_sums
