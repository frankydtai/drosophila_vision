"""Cost assembly: per-part MSE and the scaled total.

Consumes a :class:`~train.session.TrainSession` and the model forward
(``neuron.forward`` + ``neuron.readout``); produces each part's weighted-mean
contribution (``W·cost / Σ W``) and their summed total. The staged-lr loop lives in
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

Entry reduction is selected by ``cost_entry_reduce``.  ``mean_trace`` first
takes the weighted mean of model and target traces inside each fine part
(``entry_part_keys``), then compares those two means.  When a pack supplies
``gt_stds``, it additionally compares the baseline-aligned model sample SD to
``abs(a_gt) * gt_std`` with the same normalization. ``entry_sse`` is the legacy
behavior that compares every entry separately before summing and does not use
the distribution-level SD target.

Sparse cost time points (#4): ``pack.gts`` keeps the ``ms_response`` window length
(task-specific ``ms_post`` handling lives in each pack) and the subsample is gathered from both v_readout
trace and gt at cost time via ``pack.cost_ts`` (from ``cost_ms``: interval
scalar or explicit ``mss``; same times for every radius). ``gt_power`` is
recomputed on the subsample.
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
    pack_traces,
    window_time_traces,
)

from train.param import (
    SIM_DTYPE,
    override_val_from,
    params_from_z,
    val_from_enabled,
)
from train.session import TaskPack, TrainSession
from task.spread.pack import CostPartPlotSpec


COST_NORMS = ("gt_power", "a_gt2")
COST_ENTRY_REDUCTIONS = ("entry_sse", "mean_trace")


@dataclass(frozen=True)
class FusedPacks:
    """Packs with same ``i_sti`` shape / onset; one ``forward`` per fused group."""

    packs: Tuple[TaskPack, ...]
    b_offsets: Tuple[int, ...]


def pack_cost_abs_ts(pack: TaskPack, t_onset, *, entry_radius=None):
    """Absolute cost ``ts`` for sparse cost samples (or ``None``).

    Sole reader of ``cost_ts`` / ``cost_time_mask`` / ``entry_radii``.
    ``entry_radius`` is one hex-lattice radius; when set and a mask exists, keep
    that radius's columns only. Omit ``entry_radius`` → all cost samples.
    """
    if pack.cost_ts is None:
        return None
    t_onset = int(t_onset or 0)
    ts = pack.cost_ts.detach().cpu().numpy().astype(np.int64, copy=False)
    if entry_radius is None:
        return t_onset + ts
    mask = getattr(pack, "cost_time_mask", None)
    entry_radii = getattr(pack, "entry_radii", None)
    if mask is None or entry_radii is None:
        return t_onset + ts
    entry_radii = entry_radii.detach().cpu().numpy().astype(np.int64, copy=False)
    hit = np.where(entry_radii == int(entry_radius))[0]
    if not hit.size:
        return t_onset + np.zeros(0, dtype=np.int64)
    return t_onset + ts[mask[int(hit[0])].detach().cpu().numpy() > 0]


def node_vals_from_param(params, param: str, nodes, connectome, *, sim_dtype=SIM_DTYPE):
    """Per-node vals from a cell-indexed schema param (or scalar default)."""
    cell_vals = params.get(param, 1.0 if param == "a_gt" else 0.0)
    return (
        cell_vals[connectome.node_cells[nodes]]
        if torch.is_tensor(cell_vals) and cell_vals.dim() != 0 else
        torch.full(
            (int(nodes.shape[0]),),
            float(cell_vals if not torch.is_tensor(cell_vals) else cell_vals.item()),
            dtype=sim_dtype,
            device=nodes.device,
        )
    )


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
    if (
        not val_from_enabled(
            (session.train_opts if session is not None else None) or {}, "bias_gt",
        )
    ) and "v_th" in params:
        bias = bias + node_vals_from_param(params, "v_th", nodes, connectome, sim_dtype=sim_dtype)
    return a_gt, bias


def gt_affine_from_cell(
    params, cell, connectome, *, sim_dtype=SIM_DTYPE, session=None,
) -> tuple[float, float]:
    """Cell-level ``(a_gt, effective_bias)`` — mean over that cell's nodes."""
    cell_idx = [str(name) for name in connectome.cells].index(str(cell))
    a_gt, bias = gt_affine_from_nodes(
        params,
        torch.nonzero(
            connectome.node_cells == cell_idx, as_tuple=False,
        ).squeeze(-1),
        connectome,
        sim_dtype=sim_dtype,
        session=session,
    )
    return float(a_gt[0].item()), float(bias.mean().item())


def forward_pack(session, params, i_sti, pack):
    """Readout trace ``(B, T, N)`` then ``val_from`` from onset (mutates ``params``)."""
    trace = forward_trace(session, params, i_sti, pack=pack)
    override_val_from(params, session, onset_trace=trace, t_onset=pack_t_onset(pack))
    return trace


def gt_affine_from_pack(
    params,
    pack: TaskPack,
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
    return str(
        (session.train_opts or {}).get("cost_norm", TRAIN_OPTIMIZATION['cost_norm']),
    )


def _session_cost_entry_reduce(session: TrainSession) -> str:
    """Entry reduction from saved train opts (legacy runs default to entry SSE)."""
    reduction = str(
        (session.train_opts or {}).get("cost_entry_reduce", "entry_sse"),
    )
    if reduction not in COST_ENTRY_REDUCTIONS:
        raise ValueError(
            f"cost_entry_reduce must be one of {COST_ENTRY_REDUCTIONS}; "
            f"got {reduction!r}"
        )
    return reduction


def _session_a_lsd(session: TrainSession) -> float:
    """Fixed manual SD-loss weight; absent legacy runs remain mean-only."""
    value = float((session.train_opts or {}).get("a_lsd", 0.0))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"a_lsd must be finite and >= 0; got {value!r}")
    return value


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
    gt_stds: Optional[torch.Tensor] = None,
    gt_std_scales: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Local costs for parts via ``part_idxs`` (``part_idxs`` -1 = skip entry)."""
    if not part_keys:
        return {}
    if _session_cost_entry_reduce(session) == "mean_trace":
        return _mean_trace_parts_from_entries(
            a_gt, bias_gt, gts, scale, v_readout, part_idxs, part_keys,
            session, time_mask=time_mask, gt_stds=gt_stds,
            gt_std_scales=gt_std_scales,
        )
    cost_norm = _session_cost_norm(session)
    _, _, sse_wt, power_wt = _scaled_mse_terms(
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
        contrib = sse_entry / (a_gt * a_gt).clamp(min=torch.finfo(a_gt.dtype).tiny)
        costs = contrib.new_zeros((len(part_keys),))
        costs.scatter_add_(0, part_idxs_pos, contrib * keep_f)
    else:
        raise ValueError(f"cost_norm must be one of {COST_NORMS}; got {cost_norm!r}")
    parts: Dict[str, torch.Tensor] = {}
    for part_idx, part_key in enumerate(part_keys):
        if _part_scale(session, part_key) == 0.0:
            continue
        parts[part_key] = costs[part_idx]
    return parts


def _mean_trace_parts_from_entries(
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    gts: torch.Tensor,
    scale: torch.Tensor,
    v_readout: torch.Tensor,
    part_idxs: torch.Tensor,
    part_keys: List[str],
    session: TrainSession,
    time_mask: Optional[torch.Tensor] = None,
    gt_stds: Optional[torch.Tensor] = None,
    gt_std_scales: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compare weighted model/target mean and, when supplied, sample SD.

    The effective entry-weight sum is retained on the squared error.  This
    preserves existing part/radius/count weighting while removing only the
    within-part trace-dispersion penalty.  ``gt_stds`` adds an explicit SD
    target instead: model responses are first baseline-aligned with
    ``bias_gt``, then their weighted sample SD is compared with
    ``abs(a_gt) * gt_std``.  Both mean and SD errors use the same normalization.
    """
    cost_norm = _session_cost_norm(session)
    n_t = int(v_readout.shape[1])
    weights = scale[:, None].expand(-1, n_t)
    if time_mask is not None:
        weights = weights * time_mask.to(
            dtype=weights.dtype, device=weights.device,
        )
    gt_scaled = a_gt[:, None] * gts
    gt_affine = gt_scaled + bias_gt[:, None]
    tiny = torch.finfo(v_readout.dtype).tiny
    parts: Dict[str, torch.Tensor] = {}

    for part_idx, part_key in enumerate(part_keys):
        if _part_scale(session, part_key) == 0.0:
            continue
        entries = part_idxs == part_idx
        part_weights = weights[entries]
        weight_t = part_weights.sum(dim=0)
        valid_t = weight_t > 0
        safe_weight_t = weight_t.clamp(min=tiny)

        mean_v = (part_weights * v_readout[entries]).sum(dim=0) / safe_weight_t
        mean_gt_scaled = (
            (part_weights * gt_scaled[entries]).sum(dim=0) / safe_weight_t
        )
        mean_gt_affine = (
            (part_weights * gt_affine[entries]).sum(dim=0) / safe_weight_t
        )
        error = mean_v - mean_gt_affine

        std_numerator = error.new_zeros(())
        std_mean_a_gt2 = None
        if gt_stds is not None:
            std_entry_scale = (
                torch.ones_like(scale) if gt_std_scales is None else
                gt_std_scales
            )
            std_weights = (
                scale[entries, None] * std_entry_scale[entries, None]
            ).expand(-1, n_t)
            if time_mask is not None:
                std_weights = std_weights * time_mask[entries].to(
                    dtype=std_weights.dtype, device=std_weights.device,
                )
            std_weight_t = std_weights.sum(dim=0)
            std_weight2_t = (std_weights ** 2).sum(dim=0)
            std_safe_weight_t = std_weight_t.clamp(min=tiny)
            sample_denom = (
                std_weight_t - std_weight2_t / std_safe_weight_t
            )
            std_valid_t = sample_denom > 0
            responses = v_readout[entries] - bias_gt[entries, None]
            mean_response = (
                std_weights * responses
            ).sum(dim=0) / std_safe_weight_t
            model_var = (
                std_weights * (responses - mean_response) ** 2
            ).sum(dim=0) / sample_denom.clamp(min=tiny)
            # epsilon keeps sqrt's derivative finite when model variance is 0.
            model_std = torch.sqrt(
                model_var.clamp(min=0) + torch.finfo(model_var.dtype).eps
            )
            target_std = (
                std_weights
                * (a_gt[entries, None].abs() * gt_stds[entries])
            ).sum(dim=0) / std_safe_weight_t
            std_error = model_std - target_std
            std_numerator = (
                std_weight_t[std_valid_t] * std_error[std_valid_t] ** 2
            ).sum()
            std_mean_a_gt2 = (
                std_weights * (a_gt[entries, None] ** 2)
            ).sum(dim=0) / std_safe_weight_t

        if cost_norm == "gt_power":
            numerator = (
                weight_t[valid_t] * error[valid_t] ** 2
            ).sum() + _session_a_lsd(session) * std_numerator
            denominator = (
                weight_t[valid_t] * mean_gt_scaled[valid_t] ** 2
            ).sum()
            denominator = torch.where(
                denominator > 0, denominator, torch.ones_like(denominator),
            )
            cost = 100.0 * numerator / denominator
        elif cost_norm == "a_gt2":
            mean_a_gt2 = (
                part_weights * (a_gt[entries, None] ** 2)
            ).sum(dim=0) / safe_weight_t
            cost = (
                weight_t[valid_t] * error[valid_t] ** 2
                / mean_a_gt2[valid_t].clamp(min=tiny)
            ).sum()
            if std_mean_a_gt2 is not None:
                cost = cost + _session_a_lsd(session) * (
                    std_weight_t[std_valid_t] * std_error[std_valid_t] ** 2
                    / std_mean_a_gt2[std_valid_t].clamp(min=tiny)
                ).sum()
        else:
            raise ValueError(
                f"cost_norm must be one of {COST_NORMS}; got {cost_norm!r}"
            )
        parts[part_key] = cost
    return parts


def _part_scale(session: TrainSession, part_key: str) -> float:
    """Resolve an exact part scale, then its task scale; default to 1.0.

    For example, ``spread_bright_L1`` first checks that full key, then
    ``spread``.  This lets a task-wide scale coexist with finer overrides.
    """
    scales = session.part_cost_scales or {}
    if part_key in scales:
        return float(scales[part_key])
    for task in sorted(session.tasks, key=len, reverse=True):
        if part_key.startswith(f"{task}_") and task in scales:
            return float(scales[task])
    return 1.0


def _entry_cost_scales(pack: TaskPack) -> torch.Tensor:
    scales = getattr(pack, "cost_scales", None)
    if scales is not None:
        return scales
    n_cost = int(pack.entry_nodes.shape[0])
    return torch.ones(n_cost, dtype=pack.gts.dtype, device=pack.gts.device)


def entries_by_part(pack: TaskPack) -> Tuple[torch.Tensor, List[str]]:
    """``(part_idxs, part_keys)`` from ``pack.entry_part_keys``; one CPU sync."""
    n = int(pack.entry_nodes.shape[0])
    keys = pack.entry_part_keys
    if n == 0:
        if keys:
            raise ValueError(
                f"pack {pack.task!r} has entry_part_keys but no cost entries"
            )
        return torch.empty(
            0, dtype=torch.long, device=pack.entry_nodes.device,
        ), []
    if not keys or len(keys) != n:
        raise ValueError(f"pack {pack.task!r} missing entry_part_keys (n_cost={n})")
    entry_cost_scales = _entry_cost_scales(pack).detach().cpu().numpy()
    part_idxs: Dict[str, int] = {}
    part_keys: List[str] = []
    part_idxs_np = np.full(n, -1, dtype=np.int64)
    for entry in range(n):
        if entry_cost_scales[entry] <= 0.0:
            continue
        part_key = keys[entry]
        part_idx = part_idxs.get(part_key)
        if part_idx is None:
            part_idx = len(part_keys)
            part_idxs[part_key] = part_idx
            part_keys.append(part_key)
        part_idxs_np[entry] = part_idx
    return (
        torch.as_tensor(
            part_idxs_np, dtype=torch.long, device=pack.entry_nodes.device,
        ),
        part_keys,
    )


def session_cost_part_keys(session: TrainSession) -> Tuple[str, ...]:
    """Fine cost part_keys from session packs."""
    part_keys: List[str] = []
    for pack in session.iter_packs():
        part_keys.extend(entries_by_part(pack)[1])
    return tuple(part_keys)


def session_cost_part_plot_specs(
    session: TrainSession,
) -> Dict[str, CostPartPlotSpec]:
    """``part_key`` → plot grouping from session packs (no string parse)."""
    specs: Dict[str, CostPartPlotSpec] = {}
    for pack in session.iter_packs():
        if pack.cost_part_plot_specs:
            specs.update(pack.cost_part_plot_specs)
    return specs


def cost_radii_from_packs(packs, *, contrasts) -> Tuple[int, ...]:
    """Sorted cost radii from pack ``entry_radii``."""
    radii = set()
    contrast_set = set(contrasts)
    for pack in packs:
        entry_radii = getattr(pack, "entry_radii", None)
        if pack.contrast not in contrast_set or entry_radii is None:
            continue
        for radius in entry_radii.detach().cpu().numpy().tolist():
            radii.add(int(radius))
    return tuple(sorted(radii)) if radii else (0,)


def _slice_entry_part_keys(
    entry_part_keys: Tuple[str, ...],
    sel,
) -> Tuple[str, ...]:
    if not entry_part_keys:
        return ()
    sel_np = sel.detach().cpu().numpy() if torch.is_tensor(sel) else np.asarray(sel)
    if sel_np.dtype == bool:
        return tuple(
            part_key for part_key, keep in zip(entry_part_keys, sel_np) if keep
        )
    return tuple(entry_part_keys[int(entry)] for entry in sel_np.reshape(-1))


def _pack_has_active_cost(pack: TaskPack, session: TrainSession) -> bool:
    return any(
        _part_scale(session, part_key) != 0.0
        for part_key in entries_by_part(pack)[1]
    )


def _mse_entry_mask(pack: TaskPack, session: TrainSession) -> torch.Tensor:
    part_idxs, part_keys = entries_by_part(pack)
    mask = torch.zeros_like(pack.entry_bs, dtype=torch.bool)
    for part_idx, part_key in enumerate(part_keys):
        if _part_scale(session, part_key) != 0.0:
            mask |= part_idxs == part_idx
    return mask


def _pack_active_bs(pack: TaskPack, session: TrainSession) -> Tuple[int, ...]:
    """Stimulus b idxs with at least one non-zero-scale cost node."""
    entry_mask = _mse_entry_mask(pack, session)
    if not bool(entry_mask.any()):
        return ()
    return tuple(int(b) for b in pack.entry_bs[entry_mask].unique(sorted=True).tolist())


def _active_entries(
    pack: TaskPack,
    session: TrainSession,
    b: Optional[int] = None,
) -> Optional[torch.Tensor]:
    entry_mask = _mse_entry_mask(pack, session)
    if b is not None:
        entry_mask = entry_mask & (pack.entry_bs == int(b))
    if not bool(entry_mask.any()):
        return None
    return torch.nonzero(entry_mask, as_tuple=False).reshape(-1)


def _pack_entry_fields(
    pack: TaskPack,
    sel,
    *,
    entry_bs: Optional[torch.Tensor] = None,
) -> dict:
    """Indexed cost-entry tensors for ``replace``."""
    fields = {
        "gts": pack.gts[sel],
        "entry_bs": (
            pack.entry_bs[sel] if entry_bs is None else entry_bs
        ),
        "entry_nodes": pack.entry_nodes[sel],
    }
    for field in ("gt_stds", "gt_std_scales"):
        if getattr(pack, field, None) is not None:
            fields[field] = getattr(pack, field)[sel]
    cost_scales = getattr(pack, "cost_scales", None)
    if cost_scales is not None:
        fields["cost_scales"] = cost_scales[sel]
    for field in ("cost_t0s", "entry_radii", "cost_sti_us", "cost_sti_vs", "cost_pd_nds"):
        if getattr(pack, field, None) is not None:
            fields[field] = getattr(pack, field)[sel]
    if pack.entry_part_keys:
        fields["entry_part_keys"] = _slice_entry_part_keys(pack.entry_part_keys, sel)
    return fields


def _slice_pack_entries(pack: TaskPack, entries: torch.Tensor) -> TaskPack:
    return replace(pack, **_pack_entry_fields(pack, entries))


def _subset_pack_bs(pack: TaskPack, bs: Tuple[int, ...]) -> Optional[TaskPack]:
    if len(bs) == int(pack.i_sti.shape[0]):
        return pack
    device = pack.i_sti.device
    active_bs = torch.tensor(bs, dtype=torch.long, device=device)
    entry_mask = torch.isin(pack.entry_bs, active_bs)
    if not bool(entry_mask.any()):
        return None
    fields = _pack_entry_fields(
        pack,
        entry_mask,
        entry_bs=torch.full(
            (int(max(max(bs), int(pack.entry_bs.max()))) + 1,),
            -1,
            dtype=torch.long,
            device=device,
        ).scatter_(
            0,
            active_bs,
            torch.arange(len(bs), dtype=torch.long, device=device),
        )[pack.entry_bs[entry_mask]],
    )
    fields["i_sti"] = pack.i_sti.index_select(0, active_bs)
    bar_axis_distance = getattr(pack, "bar_axis_distance", None)
    if bar_axis_distance is not None:
        fields["bar_axis_distance"] = bar_axis_distance.index_select(0, active_bs)
    for field in ("i_sti_baseline_b", "i_sti_peak_b"):
        values = getattr(pack, field, None)
        if values is not None:
            fields[field] = values.index_select(0, active_bs)
    sti_bs = getattr(pack, "sti_bs", None)
    if sti_bs is not None and sti_bs.numel():
        sti_mask = torch.isin(sti_bs, active_bs)
        sti_b_map = torch.full(
            (int(max(max(bs), int(sti_bs.max()))) + 1,),
            -1,
            dtype=torch.long,
            device=device,
        ).scatter_(
            0,
            active_bs,
            torch.arange(len(bs), dtype=torch.long, device=device),
        )
        fields["sti_bs"] = sti_b_map[sti_bs[sti_mask]]
        fields["sti_nodes"] = pack.sti_nodes[sti_mask]
        for field in ("a_sti_radius_idxs", "a_sti_mid_idxs"):
            if getattr(pack, field, None) is not None:
                fields[field] = getattr(pack, field)[sti_mask]
        pulse = getattr(pack, "i_sti_pulse", None)
        if pulse is not None and pulse.dim() == 2:
            fields["i_sti_pulse"] = pulse.index_select(0, active_bs)
    return replace(pack, **fields)


def _active_cost_pack(
    pack: TaskPack,
    session: TrainSession,
    *,
    b: Optional[int] = None,
    bs: Optional[Tuple[int, ...]] = None,
) -> Optional[TaskPack]:
    """Drop zero-scale entries and, when requested, inactive sti bs."""
    if bs is not None:
        pack = _subset_pack_bs(pack, bs)
        if pack is None:
            return None
    entries = _active_entries(pack, session, b=b)
    return None if entries is None else _slice_pack_entries(pack, entries)


def _build_active_packs(session: TrainSession) -> Tuple[TaskPack, ...]:
    """Active cost entry/b packs (non-sequential mode only)."""
    if session.sequential:
        return ()
    active_packs: List[TaskPack] = []
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
    active_packs: Tuple[TaskPack, ...],
) -> Tuple[FusedPacks, ...]:
    """Group compatible packs for one forward each (cat ``i_sti`` along ``b``)."""
    if session.sequential or not active_packs:
        return ()
    grouped: Dict[Tuple, List[TaskPack]] = {}
    for pack in active_packs:
        grouped.setdefault(
            (
                int(pack.i_sti.shape[1]),
                int(pack.i_sti.shape[2]),
                str(pack.i_sti.device),
                pack.i_sti.dtype,
                pack_t_onset(pack),
                str(pack.task),
            ),
            [],
        ).append(pack)
    fused_packs: List[FusedPacks] = []
    for packs in grouped.values():
        offsets: List[int] = []
        b_offset = 0
        for pack in packs:
            offsets.append(b_offset)
            b_offset += int(pack.i_sti.shape[0])
        fused_packs.append(FusedPacks(packs=tuple(packs), b_offsets=tuple(offsets)))
    return tuple(fused_packs)


def _fused_forward_pack(fused: FusedPacks) -> TaskPack:
    """Combine drive metadata, including contrast-specific indexed pulses."""
    packs = fused.packs
    first = packs[0]
    if len(packs) == 1:
        return first
    fields = {"i_sti": torch.cat([pack.i_sti for pack in packs], dim=0)}
    if all(getattr(pack, "bar_axis_distance", None) is not None for pack in packs):
        fields["bar_axis_distance"] = torch.cat(
            [pack.bar_axis_distance for pack in packs], dim=0,
        )
        baseline_b = []
        peak_b = []
        for pack in packs:
            n_b = int(pack.i_sti.shape[0])
            baseline_b.append(torch.full(
                (n_b,), float(pack.i_sti_baseline),
                dtype=pack.i_sti.dtype, device=pack.i_sti.device,
            ))
            peak_b.append(torch.full(
                (n_b,), float(pack.i_sti_peak),
                dtype=pack.i_sti.dtype, device=pack.i_sti.device,
            ))
        fields["i_sti_baseline_b"] = torch.cat(baseline_b, dim=0)
        fields["i_sti_peak_b"] = torch.cat(peak_b, dim=0)
    idx_field = next(
        (
            field for field in ("a_sti_radius_idxs", "a_sti_mid_idxs")
            if any(getattr(pack, field, None) is not None for pack in packs)
        ),
        None,
    )
    if idx_field is None:
        return replace(first, **fields)

    sti_bs = []
    sti_nodes = []
    value_idxs = []
    pulses = []
    for pack, b_offset in zip(packs, fused.b_offsets):
        n_b = int(pack.i_sti.shape[0])
        pulse = pack.i_sti_pulse
        pulses.append(
            pulse[None, :].expand(n_b, -1) if pulse.dim() == 1 else pulse
        )
        idxs = getattr(pack, idx_field, None)
        if idxs is None:
            continue
        sti_bs.append(pack.sti_bs + int(b_offset))
        sti_nodes.append(pack.sti_nodes)
        value_idxs.append(idxs)
    fields.update({
        "i_sti_pulse": torch.cat(pulses, dim=0),
        "sti_bs": torch.cat(sti_bs, dim=0),
        "sti_nodes": torch.cat(sti_nodes, dim=0),
        idx_field: torch.cat(value_idxs, dim=0),
    })
    return replace(first, **fields)


def _pack_cost_parts_from_v_readout(
    pack: TaskPack,
    session: TrainSession,
    a_gt: torch.Tensor,
    bias_gt: torch.Tensor,
    v_readout: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    part_idxs, part_keys = entries_by_part(pack)
    if not part_keys:
        return {}
    gts = pack.gts
    gt_stds = getattr(pack, "gt_stds", None)
    if pack.cost_ts is not None:
        cost_ts = pack.cost_ts.to(device=v_readout.device)
        v_readout = v_readout.index_select(1, cost_ts)
        gts = gts.index_select(1, cost_ts)
        if gt_stds is not None:
            gt_stds = gt_stds.index_select(1, cost_ts)
    cost_time_mask = getattr(pack, "cost_time_mask", None)
    return _parts_from_entries(
        a_gt, bias_gt, gts, _entry_cost_scales(pack), v_readout,
        part_idxs, part_keys, session,
        time_mask=(
            None if cost_time_mask is None else
            cost_time_mask.to(device=v_readout.device)
        ),
        gt_stds=gt_stds,
        gt_std_scales=getattr(pack, "gt_std_scales", None),
    )


def _pack_cost_parts_from_params(params, pack: TaskPack, session: TrainSession, b=None):
    """Unscaled cost parts for one pack."""
    if b is not None:
        pack = _subset_pack_bs(pack, (int(b),))
        if pack is None:
            return {}
        b = 0
        entry_mask = pack.entry_bs == b
        trace = forward_pack(session, params, pack.i_sti, pack)
        t_onset = pack_t_onset(pack)
        n_t = int(pack.gts.shape[1])
        entry_nodes = pack.entry_nodes[entry_mask]
        cost_t0s = getattr(pack, "cost_t0s", None)
        v_readout = (
            trace[0, t_onset:t_onset + n_t, entry_nodes].transpose(0, 1)
            if cost_t0s is None else
            window_time_traces(
                trace,
                torch.zeros(
                    int(entry_mask.sum()), dtype=torch.long,
                    device=pack.entry_nodes.device,
                ),
                entry_nodes,
                cost_t0s[entry_mask],
                n_t,
                t_onset=t_onset,
            )
        )
    else:
        v_readout = pack_traces(
            forward_pack(session, params, pack.i_sti, pack), pack,
        )
    return _pack_cost_parts_from_v_readout(
        pack,
        session,
        *gt_affine_from_pack(params, pack, session, b=b),
        v_readout,
    )


def _pack_parts_span_bs(pack: TaskPack) -> bool:
    """True when any fine part has cost entries from more than one stimulus ``b``."""
    if not pack.entry_part_keys:
        return False
    part_idxs, _ = entries_by_part(pack)
    if part_idxs.numel() == 0:
        return False
    entry_bs = pack.entry_bs
    for part_idx in range(int(part_idxs.max().item()) + 1):
        mask = part_idxs == part_idx
        if not mask.any():
            continue
        if len(torch.unique(entry_bs[mask])) > 1:
            return True
    return False


def _session_parts_span_bs(session: TrainSession) -> bool:
    for pack in session.iter_packs():
        if not _pack_has_active_cost(pack, session):
            continue
        active_bs = _pack_active_bs(pack, session)
        if not active_bs:
            continue
        active_pack = _active_cost_pack(pack, session, bs=active_bs)
        if active_pack is not None and _pack_parts_span_bs(active_pack):
            return True
    return False


def _calc_unscaled_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Internal per-part cost before ``part_cost_scales`` and mean normalization."""
    params = params_from_z(z, session)
    # mean_trace over a part that spans multiple stimulus batches must see every
    # active b before reducing; otherwise per-b sequential reduction is equivalent.
    if (
        session.sequential
        and _session_cost_entry_reduce(session) == "mean_trace"
        and _session_parts_span_bs(session)
    ):
        parts: Dict[str, torch.Tensor] = {}
        for pack in session.iter_packs():
            if not _pack_has_active_cost(pack, session):
                continue
            active_bs = _pack_active_bs(pack, session)
            if not active_bs:
                continue
            active_pack = _active_cost_pack(pack, session, bs=active_bs)
            if active_pack is None:
                continue
            for part_key, part in _pack_cost_parts_from_params(
                params, active_pack, session, b=None,
            ).items():
                if _part_scale(session, part_key) != 0.0:
                    parts[part_key] = part
        return parts
    active_packs = _build_active_packs(session)
    fused_packs = _build_fused_packs(session, active_packs)
    if fused_packs:
        parts: Dict[str, torch.Tensor] = {}
        for fused in fused_packs:
            # Compatible packs share t_onset; pass one pack for prepare.
            forward_task_pack = _fused_forward_pack(fused)
            trace = forward_pack(
                session,
                params,
                forward_task_pack.i_sti,
                forward_task_pack,
            )
            for pack, b_offset in zip(fused.packs, fused.b_offsets):
                for part_key, part in _pack_cost_parts_from_v_readout(
                    pack,
                    session,
                    *gt_affine_from_pack(params, pack, session),
                    pack_traces(trace, pack, b_offset=b_offset),
                ).items():
                    if _part_scale(session, part_key) != 0.0:
                        parts[part_key] = part
        return parts
    parts: Dict[str, torch.Tensor] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    if active_packs and not session.sequential:
        for pack in active_packs:
            for part_key, part in _pack_cost_parts_from_params(
                params, pack, session, b=None,
            ).items():
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
            if _pack_has_active_cost(pack, session):
                for b in active_bs:
                    active_pack = _active_cost_pack(pack, session, b=b)
                    if active_pack is None:
                        continue
                    for part_key, part in _pack_cost_parts_from_params(
                        params, active_pack, session, b=b,
                    ).items():
                        pack_parts[part_key] = pack_parts.get(part_key, zero) + part
        else:
            active_pack = _active_cost_pack(pack, session, bs=active_bs)
            if active_pack is None:
                continue
            pack_parts = _pack_cost_parts_from_params(
                params, active_pack, session, b=None,
            )
        for part_key, part in pack_parts.items():
            if _part_scale(session, part_key) == 0.0:
                continue
            parts[part_key] = part
    return parts


def _session_part_scale_sum(session: TrainSession) -> float:
    """Σ W over discovered fine part_keys (e.g. 13·(1+6·1/6)=26 for spot center+r1)."""
    return float(sum(
        scale
        for part_key in session_cost_part_keys(session)
        if (scale := _part_scale(session, part_key)) != 0.0
    ))


def _weighted_mean_cost_parts(
    parts: Dict[str, torch.Tensor], session: TrainSession,
) -> Dict[str, torch.Tensor]:
    """Convert raw parts to objective contributions ``W·part / Σ W``.

    The returned values are directly comparable across parts and sum to the
    scalar training objective.
    """
    scale_sum = _session_part_scale_sum(session)
    if scale_sum == 0.0:
        return {}
    return {
        part_key: (_part_scale(session, part_key) / scale_sum) * part
        for part_key, part in parts.items()
        if _part_scale(session, part_key) != 0.0
    }


def calc_cost_parts(z, session: TrainSession) -> Dict[str, torch.Tensor]:
    """Per-part weighted-mean contributions; their sum equals ``calc_cost``."""
    return _weighted_mean_cost_parts(
        _calc_unscaled_cost_parts(z, session), session,
    )


def _scaled_cost_from_parts(parts: Dict[str, torch.Tensor], session: TrainSession):
    """Sum weighted-mean part contributions returned by ``calc_cost_parts``."""
    total = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    for part in parts.values():
        total = total + part
    return total


def calc_cost(z, session: TrainSession, parts=None):
    return _scaled_cost_from_parts(
        calc_cost_parts(z, session) if parts is None else parts,
        session,
    )


def _iter_cost_bs(session: TrainSession):
    """Yield ``(pack, b)`` for gradient part sums."""
    for pack in session.iter_packs():
        if not _pack_has_active_cost(pack, session):
            continue
        active_bs = _pack_active_bs(pack, session)
        if not active_bs:
            continue
        if session.sequential:
            if _pack_has_active_cost(pack, session):
                for b in active_bs:
                    active_pack = _active_cost_pack(pack, session, b=b)
                    if active_pack is not None:
                        yield active_pack, b
        else:
            active_pack = _active_cost_pack(pack, session, bs=active_bs)
            if active_pack is not None:
                yield active_pack, None


def backward_part_sums(z, session: TrainSession):
    """Backward mean local-% cost one ``b`` at a time.

    Each ``b`` contributes ``Σ (W·cost) / W_norm`` so gradients match
    ``calc_cost`` (``Σ W·cost / Σ W``). Returned ``part_sums`` are weighted
    mean contributions and therefore sum to ``total``.
    """
    if _session_cost_entry_reduce(session) == "mean_trace":
        if _session_parts_span_bs(session):
            parts = calc_cost_parts(z, session)
            total = _scaled_cost_from_parts(parts, session)
            total.backward()
            part_sums = {key: float(part.item()) for key, part in parts.items()}
            return float(total.item()), part_sums

        part_sums: Dict[str, float] = {}
        zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
        scale_norm = _session_part_scale_sum(session)
        if scale_norm == 0.0:
            return 0.0, {}
        for pack, b in _iter_cost_bs(session):
            params = params_from_z(z, session)
            scaled_cost = zero
            has_cost = False
            for part_key, part in _pack_cost_parts_from_params(
                params, pack, session, b=b,
            ).items():
                scale = _part_scale(session, part_key)
                if scale == 0.0:
                    continue
                scaled_cost = scaled_cost + (scale / scale_norm) * part
                has_cost = True
                contribution = (scale / scale_norm) * float(part.item())
                part_sums[part_key] = part_sums.get(part_key, 0.0) + contribution
            if has_cost:
                scaled_cost.backward()
        return sum(part_sums.values()), part_sums

    part_sums: Dict[str, float] = {}
    zero = torch.zeros((), dtype=session.sim_dtype, device=session.device)
    scale_norm = _session_part_scale_sum(session)
    if scale_norm == 0.0:
        return 0.0, {}
    for pack, b in _iter_cost_bs(session):
        params = params_from_z(z, session)
        scaled_cost = zero
        has_cost = False
        for part_key, part in _pack_cost_parts_from_params(
            params, pack, session, b=b,
        ).items():
            scale = _part_scale(session, part_key)
            if scale == 0.0:
                continue
            scaled_cost = scaled_cost + (scale / scale_norm) * part
            has_cost = True
            contribution = (scale / scale_norm) * float(part.item())
            part_sums[part_key] = part_sums.get(part_key, 0.0) + contribution
        if has_cost:
            scaled_cost.backward()
    return (
        sum(part_sums.values())
    ), part_sums
