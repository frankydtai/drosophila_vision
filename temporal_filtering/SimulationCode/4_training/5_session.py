# -*- coding: utf-8 -*-
"""Session assembly: build a :class:`TrainSession` from canonical training opts.

Owns stimulus-opts finalisation (CLI overrides -> per-target sidecar dicts),
network backend construction, and the per-target ``TargetPack`` builders. The
builders wrap the neutral target dataclasses from ``task`` (which sit below
``training`` in the import graph) and stamp the cross-cutting readout controls:

* spot: ``readout_kind`` (#2), sparse ``cost_time_ix`` (#4), pulse ``pulse_ms``
  (#1) already baked into the stimulus, ``always_waveform_mse=True``,
  ``signal_scale`` (hp_lp peak PR);
* moving bar: ``always_waveform_mse=False`` (waveform MSE only when a cost
  window exists), ``signal_scale``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from network.construction import I_BASELINE, I_BRIGHT, I_DARK
from network.connectivity import SIM_DTYPE_DEFAULT, sim_dtype_from_fp32
from neuron.params import DELTA_MS, ms_to_t
from training.config import run_data_dir
from neuron import (
    IH_OFF_DEFAULT,
    SYN_MODE_DEFAULT,
    borst_schema,
    default_schema,
    exc_synweight,
    inh_synweight,
    normalize_syn_mode,
)

from training.config import (
    CLI_TARGET_NAMES,
    I_CLI_BRIGHT_TARGETS,
    I_CLI_DARK_TARGETS,
    I_CLI_SIDECAR_FIELD,
    PD_ND_LABELS,
    SPOT_TARGETS,
    TARGET_ALIASES,
    TARGET_I_FIELDS,
    TRAIN_OPTS_FILE,
    VALID_TARGETS,
    _SPOT_STEP_KEY,
    expand_cost_weight_dict,
    moving_bar_cost_part_key,
    normalize_target_list,
)
from training.cost import _build_cost_subpacks, _build_fused_borst
from training.params import (
    apply_partitions,
    attach_param_carry,
    build_e_leak,
    build_ih_dir,
    schema_nparams,
    schema_partitions_record,
    unit_names_for_segment,
)
from training.target_pack import ModelBackend, TargetPack, TrainSession, active_device

from task.spot.data import (
    SPOT_POLARITIES,
    build_shifted_target,
    expand_spot_cost_r_w_dict,
    make_spot_stimulus_opts,
)
from task.spot.input import (
    DEFAULT_FULLY_INSIDE,
    DEFAULT_MULTI_SPOT,
    DEFAULT_SHIFT_EXTENT,
    DEFAULT_SPOT_EXTENT,
    spot_extent_half_steps,
    spot_timing_t_from_opts,
)
from task.moving_bar.data import (
    _enrich_moving_bar_stimulus_opts,
    _readout_subtypes_from_opts,
    build_moving_bar_target,
    make_moving_bar_stimulus_opts,
)
from task.moving_bar.input import resolve_i_baseline
from network.construction import load_network, unit_type_names
from network.layout import normalize_cost_extent


def _opt_float(opts, *keys, default=None):
    for key in keys:
        if key in opts:
            return float(opts[key])
    if default is not None:
        return float(default)
    raise KeyError(f"expected one of {keys!r} in stimulus opts")


def _spot_i_from_opts(opts, polarity: str):
    """Read spot PR currents (``i_baseline`` / bright or dark step)."""
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    step_key = _SPOT_STEP_KEY[polarity]
    step_default = I_BRIGHT if polarity == "bright" else I_DARK
    return (
        _opt_float(opts, "i_baseline", default=I_BASELINE),
        _opt_float(opts, step_key, default=step_default),
    )


def _signal_scale_from_opts(pack_name: str, opts: Optional[dict]) -> float:
    """Peak PR current for hp_lp ``sig / scale``; stamped onto ``TargetPack``."""
    opts = opts or {}
    if pack_name == "spot_bright":
        peak = _opt_float(opts, "i_bright", default=I_BRIGHT)
    elif pack_name == "spot_dark":
        peak = _opt_float(opts, "i_dark", default=I_DARK)
    elif pack_name == "moving_bar_bright":
        peak = _opt_float(opts, "i_bright_bar", default=I_BRIGHT)
    elif pack_name == "moving_bar_dark":
        peak = _opt_float(opts, "i_dark_bar", default=I_DARK)
    else:
        peak = I_BRIGHT
    peak = float(peak)
    if peak == 0.0:
        return float(I_BRIGHT)
    return peak


def resolve_type_indices(type_names, backend: ModelBackend):
    """Map cell-type names to indices in the network vocabulary."""
    if backend.network is None:
        raise ValueError("resolve_type_indices requires backend.network")
    names = [str(n) for n in type_names]
    tn = list(backend.network.type_names)
    return [tn.index(n) for n in names if n in tn]


def extend_target_pack_mirror_fit(pack, mirror_types, mirror_fit, mirror_sign=-1.0, backend=None):
    """Extend a :class:`TargetPack`: mirror *mirror_fit* targets onto *mirror_types*."""
    if backend is None or backend.network is None:
        raise ValueError("extend_target_pack_mirror_fit requires backend.network")
    return _extend_pack_mirror_fit_network(
        pack, mirror_types, mirror_fit, mirror_sign, backend.network,
    )


def _extend_pack_mirror_fit_network(pack, mirror_types, mirror_fit, mirror_sign, C):
    names = unit_type_names(C)
    u_arr = pack.readout_unit.cpu().numpy()
    b_arr = pack.readout_batch.cpu().numpy()
    w_arr = pack.cost_weight.cpu().numpy()
    r_arr = (
        pack.cost_radius.cpu().numpy()
        if pack.cost_radius is not None else None
    )
    col_u_all = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    col_v_all = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    extra_b, extra_u, extra_rows, extra_w, extra_r, extra_pd_nd = [], [], [], [], [], []
    for row_i in range(len(u_arr)):
        u = int(u_arr[row_i])
        if str(names[u]) != mirror_fit:
            continue
        b = int(b_arr[row_i])
        col_u, col_v = int(col_u_all[u]), int(col_v_all[u])
        mirror_target = float(mirror_sign) * pack.data[row_i:row_i + 1]
        w = float(w_arr[row_i])
        r = float(r_arr[row_i]) if r_arr is not None else None
        for mtype in mirror_types:
            candidates = np.where(
                (col_u_all == col_u)
                & (col_v_all == col_v)
                & (names == str(mtype))
            )[0]
            for uidx in candidates:
                extra_b.append(b)
                extra_u.append(int(uidx))
                extra_rows.append(mirror_target)
                extra_w.append(w)
                if r is not None:
                    extra_r.append(r)
                if pack.cost_pd_nd is not None:
                    extra_pd_nd.append(int(pack.cost_pd_nd[row_i].item()))
    return _append_mirror_pack_rows(
        pack, extra_u, extra_rows,
        readout_batch=extra_b, cost_weight=extra_w,
        cost_radius=extra_r if extra_r else None,
        cost_pd_nd=extra_pd_nd if extra_pd_nd else None,
    )


def _append_mirror_pack_rows(
    pack, extra_units, extra_rows, readout_batch=None, cost_weight=None, cost_radius=None,
    cost_pd_nd=None,
):
    extra_units_t = torch.tensor(extra_units, dtype=torch.long, device=active_device())
    extra_data_t = torch.cat(extra_rows, dim=0)
    n_all = int(pack.readout_unit.shape[0]) + len(extra_units)
    n_extra = len(extra_units)
    if readout_batch is None:
        readout_batch = torch.zeros(n_extra, dtype=torch.long, device=active_device())
    else:
        readout_batch = torch.tensor(readout_batch, dtype=torch.long, device=active_device())
    if cost_weight is None:
        w_dtype = pack.cost_weight.dtype
        cost_weight = torch.ones(n_all, dtype=w_dtype, device=active_device())
    else:
        base_w = pack.cost_weight
        w_dtype = base_w.dtype
        cost_weight = torch.cat([
            base_w,
            torch.tensor(cost_weight, dtype=w_dtype, device=active_device()),
        ])
    cost_radius_out = pack.cost_radius
    if cost_radius is not None:
        base_r = pack.cost_radius
        r_dtype = base_r.dtype if base_r is not None else SIM_DTYPE_DEFAULT
        extra_r_t = torch.tensor(cost_radius, dtype=r_dtype, device=active_device())
        cost_radius_out = (
            torch.cat([base_r, extra_r_t])
            if base_r is not None else extra_r_t
        )
    all_data = torch.cat([pack.data, extra_data_t], dim=0)
    cost_pd_nd_out = pack.cost_pd_nd
    if cost_pd_nd is not None:
        extra_pd_t = torch.tensor(cost_pd_nd, dtype=torch.long, device=active_device())
        cost_pd_nd_out = (
            torch.cat([pack.cost_pd_nd, extra_pd_t])
            if pack.cost_pd_nd is not None else extra_pd_t
        )
    return TargetPack(
        name=pack.name,
        signal=pack.signal,
        data=all_data,
        power=pack.power + torch.sum(extra_data_t ** 2),
        cost_weight=cost_weight,
        readout_batch=torch.cat([pack.readout_batch, readout_batch]),
        readout_unit=torch.cat([pack.readout_unit, extra_units_t]),
        cost_t0=pack.cost_t0,
        cost_radius=cost_radius_out,
        cost_extent=pack.cost_extent,
        cost_pd_nd=cost_pd_nd_out,
        dsi_pos_rows=pack.dsi_pos_rows,
        dsi_neg_rows=pack.dsi_neg_rows,
        dsi_pos_ptr=pack.dsi_pos_ptr,
        dsi_neg_ptr=pack.dsi_neg_ptr,
        dsi_target=pack.dsi_target,
        dsi_weight=pack.dsi_weight,
        dsi_power=pack.dsi_power,
        readout_kind=pack.readout_kind,
        cost_time_ix=pack.cost_time_ix,
        always_waveform_mse=pack.always_waveform_mse,
        signal_scale=pack.signal_scale,
    )


def _mirror_types_from_spec(spec):
    if "mirror_types" not in spec:
        raise ValueError(f"mirror_fit spec needs mirror_types: {spec!r}")
    return [str(t) for t in spec["mirror_types"]]


def _apply_mirror_fit_spec(pack, spec, backend: ModelBackend):
    return extend_target_pack_mirror_fit(
        pack,
        mirror_types=_mirror_types_from_spec(spec),
        mirror_fit=spec["mirror_fit"],
        mirror_sign=float(spec.get("mirror_sign", -1.0)),
        backend=backend,
    )


def apply_pack_override(pack, override, backend: ModelBackend):
    """Apply one serializable pack override dict (saved in ``train_opts.json``)."""
    if "mirror_fits" in override:
        for spec in override["mirror_fits"]:
            pack = _apply_mirror_fit_spec(pack, spec, backend)
        return pack
    if "mirror_fit" in override:
        return _apply_mirror_fit_spec(pack, override["mirror_fit"], backend)
    raise ValueError(f"unknown pack override {override!r}")


def _network_backend_from_connectome(C, *, sim_dtype=SIM_DTYPE_DEFAULT) -> ModelBackend:
    """Build a :class:`ModelBackend` from an already-loaded connectome graph."""
    from neuron.params import LEAK_DEPOL_TYPES

    tn = list(C.type_names)
    depol = tuple(tn.index(t) for t in LEAK_DEPOL_TYPES if t in tn)
    conn = C.conn
    return ModelBackend(
        conn=conn,
        e_leak=build_e_leak(conn, C.n_types, depol_cells=depol, dtype=sim_dtype),
        ih_dir=build_ih_dir(conn, dtype=sim_dtype),
        n_types=C.n_types,
        n_cols=1,
        network=C,
        depol_cells=depol,
    )


def load_network_backend(
    network_json,
    dev: Optional[str] = None,
    *,
    sim_dtype=SIM_DTYPE_DEFAULT,
    syn_mode=SYN_MODE_DEFAULT,
) -> ModelBackend:
    """Load connectome network into a :class:`ModelBackend`."""
    dev = dev or active_device()
    mode = normalize_syn_mode(syn_mode)
    C = load_network(
        network_json, device=dev,
        exc_synweight=exc_synweight, inh_synweight=inh_synweight,
        dtype=sim_dtype, syn_mode=mode,
    )
    backend = _network_backend_from_connectome(C, sim_dtype=sim_dtype)
    print(f"network: {network_json}")
    print(f"  n_units={backend.n_units}, n_types={backend.n_types}, "
          f"n_pairs={backend.conn.n_pairs}, n_edges={backend.conn.n_edges}, "
          f"syn_mode={mode}, "
          f"nparams={schema_nparams(default_schema('borst', backend, syn_mode=mode))}")
    return backend


@dataclass
class _TrainBindCtx:
    """Per-target builder context during :func:`open_session`."""

    model_backend: ModelBackend
    dev: str
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT
    cost_weights: Optional[Dict[str, float]] = None
    spot_bright_stimulus_opts: Optional[dict] = None
    spot_dark_stimulus_opts: Optional[dict] = None
    moving_bar_bright_stimulus_opts: Optional[dict] = None
    moving_bar_dark_stimulus_opts: Optional[dict] = None


def _moving_bar_waveform_mse_enabled(cost_weights: Optional[dict], pack_name: str) -> bool:
    """True if PD or ND waveform MSE weight is non-zero for ``pack_name``."""
    w = expand_cost_weight_dict(cost_weights or {})
    return any(
        float(w.get(moving_bar_cost_part_key(pack_name, lab), 1.0)) != 0.0
        for lab in PD_ND_LABELS
    )


def _moving_bar_polarity_opts(ctx: _TrainBindCtx, polarity: str) -> dict:
    if polarity == "bright":
        raw = ctx.moving_bar_bright_stimulus_opts
    elif polarity == "dark":
        raw = ctx.moving_bar_dark_stimulus_opts
    else:
        raise ValueError(f"unknown moving-bar polarity {polarity!r}")
    if raw:
        return dict(raw)
    return make_moving_bar_stimulus_opts(polarity)


def _cost_extent_column_coltag(cost_extent, n_cost_columns) -> str:
    extent_tag = "all columns" if cost_extent is None else f"extent={int(cost_extent)}"
    if isinstance(n_cost_columns, dict):
        cols = ", ".join(
            f"b{int(batch)}={int(count)}"
            for batch, count in sorted(n_cost_columns.items())
        )
        return f"cost columns per batch [{cols}], {extent_tag}"
    return f"{int(n_cost_columns)} cost columns, {extent_tag}"


def _build_network_moving_bar_target(ctx: _TrainBindCtx, C, *, pack_name: str, polarity: str):
    dev = ctx.dev or active_device()
    opts = _moving_bar_polarity_opts(ctx, polarity)
    if opts.get("mode") != "network":
        opts = dict(opts)
        opts["mode"] = "network"
    if "cost_extent" in opts:
        cost_extent = normalize_cost_extent(opts["cost_extent"])
    else:
        network_extent = int(C.meta.get("extent", -1))
        default_extent = -1 if network_extent <= 0 else network_extent - 1
        cost_extent = normalize_cost_extent(default_extent)
    build_kw = dict(
        C=C,
        device=dev,
        sim_dtype=ctx.sim_dtype,
        t_on=ms_to_t(float(opts["pre_ms"]), delta_ms=float(opts.get("delta_ms", DELTA_MS))),
        cost_extent=cost_extent,
        i_baseline=opts["i_baseline"],
        contrasts=(polarity,),
        readout_subtypes=_readout_subtypes_from_opts(opts),
        multi_bar=bool(opts.get("multi_bar", True)),
        waveform_mse=_moving_bar_waveform_mse_enabled(ctx.cost_weights, pack_name),
    )
    if polarity == "bright":
        build_kw["i_bright_bar"] = opts["i_bright_bar"]
    else:
        build_kw["i_dark_bar"] = opts["i_dark_bar"]
    T = build_moving_bar_target(**build_kw)
    stim = _enrich_moving_bar_stimulus_opts(opts, T.info, cost_extent=cost_extent)
    pack = TargetPack(
        name=pack_name,
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=T.cost_t0,
        cost_extent=cost_extent,
        cost_pd_nd=T.cost_pd_nd,
        dsi_pos_rows=T.dsi_pos_rows,
        dsi_neg_rows=T.dsi_neg_rows,
        dsi_pos_ptr=T.dsi_pos_ptr,
        dsi_neg_ptr=T.dsi_neg_ptr,
        dsi_target=T.dsi_target,
        dsi_weight=T.dsi_weight,
        dsi_power=T.dsi_power,
        always_waveform_mse=False,
        signal_scale=_signal_scale_from_opts(pack_name, opts),
    )
    coltag = _cost_extent_column_coltag(cost_extent, T.info["n_cost_columns"])
    tag = (
        f"moving-bar {polarity} (B={T.n_batch} stimuli, "
        f"{T.info['n_cost']} cost cells, {coltag})"
    )
    return pack, stim, tag


def _build_network_moving_bar_bright_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    return _build_network_moving_bar_target(
        ctx, C, pack_name="moving_bar_bright", polarity="bright",
    )


def _build_network_moving_bar_dark_target(
    ctx: _TrainBindCtx, C,
) -> Tuple[TargetPack, dict, str]:
    return _build_network_moving_bar_target(
        ctx, C, pack_name="moving_bar_dark", polarity="dark",
    )


def _spot_cost_times_ms(opts):
    """Post-onset cost times from ``cost_interval_ms``: 0, interval, 2*interval, ... to last sample."""
    interval_ms = opts.get("cost_interval_ms")
    if interval_ms is None:
        return None
    interval_ms = float(interval_ms)
    if interval_ms <= 0:
        raise ValueError("cost_interval_ms must be > 0")
    delta_ms = float(opts.get("delta_ms", DELTA_MS))
    t_on, n_t = spot_timing_t_from_opts(opts)
    post = n_t - t_on
    if post <= 0:
        raise ValueError("spot post-onset window must be > 0 for cost_interval_ms")
    interval_t = max(1, int(round(interval_ms / delta_ms)))
    t_list = list(range(0, post, interval_t))
    if not t_list:
        end_ms = (post - 1) * delta_ms
        raise ValueError(
            f"cost_interval_ms={interval_ms} exceeds post-onset window "
            f"({end_ms:g} ms, post n_t={post})"
        )
    return [t * delta_ms for t in t_list]


def _spot_cost_time_ix(opts, *, device):
    """#4: sparse post-onset t indices from ``cost_interval_ms`` (relative to onset)."""
    cost_time_ms = _spot_cost_times_ms(opts)
    if not cost_time_ms:
        return None
    delta_ms = float(opts.get("delta_ms", DELTA_MS))
    t_on, n_t = spot_timing_t_from_opts(opts)
    post = n_t - t_on
    ix = [int(round(float(ms) / delta_ms)) for ms in cost_time_ms]
    bad = [ms for ms, t in zip(cost_time_ms, ix) if t < 0 or t >= post]
    if bad:
        raise ValueError(
            f"cost_interval_ms -> {bad[0]} ms post-onset t out of range [0,{post})"
        )
    ix = sorted(set(ix))
    return torch.tensor(ix, dtype=torch.long, device=device)


def _build_network_spot_target(
    ctx: _TrainBindCtx, C, *, polarity: str,
) -> Tuple[TargetPack, dict, str]:
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    pack_name = f"spot_{polarity}"
    step_key = _SPOT_STEP_KEY[polarity]
    ctx_opts = (
        ctx.spot_bright_stimulus_opts if polarity == "bright" else ctx.spot_dark_stimulus_opts
    )
    opts = dict(ctx_opts or make_spot_stimulus_opts(polarity, mode="network"))
    cost_extent = normalize_cost_extent(opts.get("cost_extent"))
    shift_extent = int(opts.get("shift_extent", DEFAULT_SHIFT_EXTENT))
    spot_extent = float(opts.get("spot_extent", DEFAULT_SPOT_EXTENT))
    multi_spot = bool(opts.get("multi_spot", DEFAULT_MULTI_SPOT))
    fully_inside = bool(opts.get("fully_inside", DEFAULT_FULLY_INSIDE))
    dev = ctx.dev or active_device()
    readout_kind = str(opts.get("filter", "ca"))
    t_on, n_t = spot_timing_t_from_opts(opts)
    T = build_shifted_target(
        C,
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        shift_extent=shift_extent,
        device=dev,
        sim_dtype=ctx.sim_dtype,
        n_t=n_t,
        t_on=t_on,
        cost_extent=cost_extent,
        spot_cost_radius_weight=expand_spot_cost_r_w_dict(stimulus_opts=opts),
        i_baseline=opts["i_baseline"],
        polarity=polarity,
        pulse_ms=opts.get("pulse_ms"),
        readout_kind=readout_kind,
        **{step_key: opts[step_key]},
    )
    cost_time_ix = _spot_cost_time_ix(opts, device=dev)
    stim = dict(opts)
    pack = TargetPack(
        name=pack_name,
        signal=T.signal,
        data=T.data,
        power=T.power,
        cost_weight=T.cost_weight,
        readout_batch=T.readout_batch,
        readout_unit=T.readout_unit,
        cost_t0=None,
        cost_radius=T.cost_radius,
        readout_stim_u=T.readout_stim_u,
        readout_stim_v=T.readout_stim_v,
        cost_extent=cost_extent,
        readout_kind=readout_kind,
        cost_time_ix=cost_time_ix,
        always_waveform_mse=True,
        signal_scale=_signal_scale_from_opts(pack_name, opts),
    )
    coltag = _cost_extent_column_coltag(cost_extent, T.info["n_cost_columns"])
    shifttag = f"{T.info['n_shifts']} shifts"
    tag = (
        f"{pack_name} (B={T.n_batch} stimuli [{T.info['n_centers']} centers simultaneous "
        f"x {shifttag}], {T.info['n_cost']} cost cells, {coltag})"
    )
    return pack, stim, tag


NETWORK_TARGET_BUILDERS = {
    "spot_bright": lambda ctx, C: _build_network_spot_target(ctx, C, polarity="bright"),
    "spot_dark": lambda ctx, C: _build_network_spot_target(ctx, C, polarity="dark"),
    "moving_bar_bright": _build_network_moving_bar_bright_target,
    "moving_bar_dark": _build_network_moving_bar_dark_target,
}


def apply_cost_extent_to_stimulus_opts(opts, target_name, cost_extent_by_target):
    """Set an explicitly resolved ``cost_extent`` on one target's options."""
    out = dict(opts or {})
    if cost_extent_by_target and target_name in cost_extent_by_target:
        out["cost_extent"] = int(cost_extent_by_target[target_name])
    elif "cost_extent" in out:
        if out["cost_extent"] is None:
            out.pop("cost_extent", None)
        else:
            out["cost_extent"] = int(out["cost_extent"])
    return out


def apply_shift_extent_to_stimulus_opts(opts, target_name, shift_extent):
    """Set ``shift_extent`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["shift_extent"] = int(shift_extent)
    return out


def apply_spot_extent_to_stimulus_opts(opts, target_name, spot_extent):
    """Set ``spot_extent`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    spot_extent_half_steps(spot_extent)
    out = dict(opts or {})
    out["spot_extent"] = float(spot_extent)
    return out


def apply_multi_spot_to_stimulus_opts(opts, target_name, multi_spot):
    """Set ``multi_spot`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["multi_spot"] = bool(multi_spot)
    return out


def apply_fully_inside_to_stimulus_opts(opts, target_name, fully_inside):
    """Set ``fully_inside`` on spot stimulus opts."""
    if target_name not in SPOT_TARGETS:
        return opts
    out = dict(opts or {})
    out["fully_inside"] = bool(fully_inside)
    return out


def apply_spot_cost_radius_weight_to_stimulus_opts(opts, target_name, spot_cost_radius_weight):
    """Set ``spot_cost_radius_weight`` on spot stimulus opts (``None`` -> default weights)."""
    if target_name not in SPOT_TARGETS or spot_cost_radius_weight is None:
        return opts
    out = dict(opts or {})
    out["spot_cost_radius_weight"] = {
        str(k): float(v) for k, v in spot_cost_radius_weight.items()
    }
    return out


def _i_cli_target_names(cli_field, name):
    """Resolve CLI target token for one ``--i_*`` flag."""
    if name not in CLI_TARGET_NAMES:
        raise ValueError(
            f"unknown target {name!r} in --{cli_field} "
            f"(expected {'|'.join(CLI_TARGET_NAMES)})",
        )
    if cli_field == "i_baseline":
        if name in TARGET_ALIASES:
            return TARGET_ALIASES[name]
        return [name]
    if cli_field == "i_bright":
        if name not in I_CLI_BRIGHT_TARGETS:
            raise ValueError(
                f"--i-bright does not accept target {name!r} "
                f"(expected spot|spot_bright|moving_bar|moving_bar_bright)",
            )
        return list(I_CLI_BRIGHT_TARGETS[name])
    if name not in I_CLI_DARK_TARGETS:
        raise ValueError(
            f"--i-dark does not accept target {name!r} "
            f"(expected spot|spot_dark|moving_bar|moving_bar_dark)",
        )
    return list(I_CLI_DARK_TARGETS[name])


def build_i_cli_by_target(kv_by_field):
    """Merge per-flag comma KV dicts into ``{'by_target': {target: {field: val}}}``."""
    by_target = {}
    for cli_field, kv in kv_by_field.items():
        if not kv:
            continue
        for name, val in kv.items():
            for t in _i_cli_target_names(cli_field, name):
                sidecar_field = I_CLI_SIDECAR_FIELD[(cli_field, t)]
                by_target.setdefault(t, {})[sidecar_field] = float(val)
    return {"by_target": by_target} if by_target else None


def apply_i_cli_to_stimulus_opts(opts, target_name, i_cli):
    """Merge per-target CLI ``--i_*`` overrides into stimulus opts."""
    if not i_cli:
        return opts
    overrides = (i_cli.get("by_target") or {}).get(target_name)
    if not overrides:
        return opts
    out = dict(opts or {})
    allowed = TARGET_I_FIELDS[target_name]
    for key, val in overrides.items():
        if key not in allowed:
            raise ValueError(f"{key!r} not valid for target {target_name!r}")
        out[key] = float(val)
    return out


_STIMULUS_TRAIN_OPT_SPECS = (
    ("spot_bright", "spot_bright_stimulus_opts"),
    ("spot_dark", "spot_dark_stimulus_opts"),
    ("moving_bar_bright", "moving_bar_bright_stimulus_opts"),
    ("moving_bar_dark", "moving_bar_dark_stimulus_opts"),
)


def _finalize_stimulus_opts(
    opts,
    target_name,
    *,
    session_mode=None,
    cost_extent_by_target,
    shift_extent,
    spot_extent,
    multi_spot,
    fully_inside,
    spot_cost_radius_weight,
    i_cli,
):
    build_mode = session_mode if session_mode is not None else (opts or {}).get("mode", "network")
    if target_name in SPOT_TARGETS:
        polarity = "bright" if target_name == "spot_bright" else "dark"
        step_key = _SPOT_STEP_KEY[polarity]
        out = make_spot_stimulus_opts(polarity, mode=build_mode, **{
            k: v for k, v in (opts or {}).items()
            if k in (
                "i_baseline", step_key, "shift_extent", "spot_extent",
                "multi_spot", "fully_inside", "pre_ms", "response_ms", "delta_ms",
                "pulse_ms", "cost_interval_ms", "filter",
            )
        })
    elif target_name == "moving_bar_bright":
        out = make_moving_bar_stimulus_opts(
            "bright",
            mode=build_mode,
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_bright_bar", "readout_subtypes", "multi_bar",
                         "pre_ms", "delta_ms")
            },
        )
    elif target_name == "moving_bar_dark":
        out = make_moving_bar_stimulus_opts(
            "dark",
            mode=build_mode,
            **{
                k: v for k, v in (opts or {}).items()
                if k in ("i_baseline", "i_dark_bar", "readout_subtypes", "multi_bar",
                         "pre_ms", "delta_ms")
            },
        )
    else:
        out = dict(opts or {})
    out = apply_cost_extent_to_stimulus_opts(out, target_name, cost_extent_by_target)
    out = apply_shift_extent_to_stimulus_opts(out, target_name, shift_extent)
    out = apply_spot_extent_to_stimulus_opts(out, target_name, spot_extent)
    out = apply_multi_spot_to_stimulus_opts(out, target_name, multi_spot)
    out = apply_fully_inside_to_stimulus_opts(out, target_name, fully_inside)
    out = apply_spot_cost_radius_weight_to_stimulus_opts(out, target_name, spot_cost_radius_weight)
    out = apply_i_cli_to_stimulus_opts(out, target_name, i_cli)
    if session_mode is not None:
        out["mode"] = session_mode
    return out


def make_train_opts(
    backend="network",
    target_list=None,
    cost_weights=None,
    pack_overrides=None,
    sequential=None,
    cost_extent_by_target=None,
    shift_extent=None,
    spot_extent=None,
    multi_spot=True,
    fully_inside=True,
    spot_cost_radius_weight=None,
    i_cli=None,
    moving_bar_bright_stimulus_opts=None,
    moving_bar_dark_stimulus_opts=None,
    spot_bright_stimulus_opts=None,
    spot_dark_stimulus_opts=None,
    network_json=None,
    network=None,
    param_partitions=None,
    syn_mode=SYN_MODE_DEFAULT,
    dev=None,
    packs=None,
    ih_off=IH_OFF_DEFAULT,
    fp32=False,
):
    """Canonical training opts for :func:`open_session` (network backend)."""
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    if network is None and network_json is None:
        raise ValueError("make_train_opts requires network or network_json")
    tl = normalize_target_list(target_list)
    mode = "network"
    if spot_extent is None:
        spot_extent = DEFAULT_SPOT_EXTENT
    if shift_extent is None:
        shift_extent = DEFAULT_SHIFT_EXTENT
    raw_by_name = {
        "spot_bright": spot_bright_stimulus_opts,
        "spot_dark": spot_dark_stimulus_opts,
        "moving_bar_bright": moving_bar_bright_stimulus_opts,
        "moving_bar_dark": moving_bar_dark_stimulus_opts,
    }
    finalize_kw = dict(
        cost_extent_by_target=cost_extent_by_target,
        shift_extent=shift_extent,
        spot_extent=spot_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
        spot_cost_radius_weight=spot_cost_radius_weight,
        i_cli=i_cli,
    )
    stimulus_opts = {}
    for tname, opts_key in _STIMULUS_TRAIN_OPT_SPECS:
        raw = raw_by_name[tname]
        if tname not in tl and raw is None:
            stimulus_opts[opts_key] = None
            continue
        stimulus_opts[opts_key] = _finalize_stimulus_opts(
            raw,
            tname,
            session_mode=mode if tname in tl else None,
            **finalize_kw,
        )
    opts = {
        "backend": "network",
        "target_list": tl,
        "cost_weights": expand_cost_weight_dict(cost_weights or {}),
        "sequential": sequential,
        **stimulus_opts,
    }
    if pack_overrides is not None:
        opts["pack_overrides"] = pack_overrides
    if packs is not None:
        opts["packs"] = packs
    if param_partitions is not None:
        opts["param_partitions"] = param_partitions
    opts["ih_off"] = str(ih_off)
    opts["syn_mode"] = normalize_syn_mode(syn_mode)
    if fp32:
        opts["fp32"] = True
    opts.update({
        "network": network,
        "network_json": str(network_json) if network_json is not None else None,
        "dev": dev,
    })
    return opts


def _train_opts_for_sidecar(
    opts, backend, target_list,
    resolved_spot_bright, resolved_spot_dark,
    resolved_bar_bright, resolved_bar_dark, sequential_bool,
) -> dict:
    record = {
        "backend": str(backend),
        "target_list": list(target_list),
        "cost_weights": {str(k): float(v) for k, v in (opts.get("cost_weights") or {}).items()},
        "sequential": bool(sequential_bool),
    }
    if backend == "network":
        record.update({
            "network_json": str(opts["network_json"]),
            "spot_bright_stimulus_opts": (
                resolved_spot_bright if resolved_spot_bright is not None
                else opts.get("spot_bright_stimulus_opts")
            ),
            "spot_dark_stimulus_opts": (
                resolved_spot_dark if resolved_spot_dark is not None
                else opts.get("spot_dark_stimulus_opts")
            ),
            "moving_bar_bright_stimulus_opts": (
                resolved_bar_bright if resolved_bar_bright is not None
                else opts.get("moving_bar_bright_stimulus_opts")
            ),
            "moving_bar_dark_stimulus_opts": (
                resolved_bar_dark if resolved_bar_dark is not None
                else opts.get("moving_bar_dark_stimulus_opts")
            ),
        })
    else:
        record["moving_bar_bright_stimulus_opts"] = (
            resolved_bar_bright if resolved_bar_bright is not None
            else opts.get("moving_bar_bright_stimulus_opts")
        )
        record["moving_bar_dark_stimulus_opts"] = (
            resolved_bar_dark if resolved_bar_dark is not None
            else opts.get("moving_bar_dark_stimulus_opts")
        )
        record["spot_bright_stimulus_opts"] = (
            resolved_spot_bright if resolved_spot_bright is not None
            else opts.get("spot_bright_stimulus_opts")
        )
        record["spot_dark_stimulus_opts"] = (
            resolved_spot_dark if resolved_spot_dark is not None
            else opts.get("spot_dark_stimulus_opts")
        )
    overrides = opts.get("pack_overrides")
    if overrides:
        record["pack_overrides"] = overrides
    if opts.get("param_partitions"):
        record["param_partitions"] = opts["param_partitions"]
    if "ih_off" in opts:
        record["ih_off"] = str(opts["ih_off"])
    record["syn_mode"] = normalize_syn_mode(opts.get("syn_mode", SYN_MODE_DEFAULT))
    if opts.get("fp32"):
        record["fp32"] = True
    return record


def _schema_from_opts(model, model_backend, schema, train_opts_record):
    if schema is not None:
        return list(schema)
    syn_mode = SYN_MODE_DEFAULT
    if train_opts_record:
        syn_mode = normalize_syn_mode(train_opts_record.get("syn_mode", SYN_MODE_DEFAULT))
    base = default_schema(model, model_backend, syn_mode=syn_mode)
    if not train_opts_record:
        return base
    parts = train_opts_record.get("param_partitions")
    if parts:
        base = apply_partitions(
            base, parts, lambda seg: unit_names_for_segment(seg, model_backend),
        )
    return base


def _make_session(
    model_backend: ModelBackend,
    model: str,
    target_list: List[str],
    packs: Dict[str, TargetPack],
    *,
    cost_weights=None,
    sequential=None,
    dev=None,
    train_opts_record=None,
    schema: Optional[list] = None,
    sim_dtype=SIM_DTYPE_DEFAULT,
) -> TrainSession:
    dev_ref = dev or active_device()
    seq = False if sequential is None else bool(sequential)
    if train_opts_record is not None:
        train_opts_record["model"] = model
        train_opts_record["sequential"] = bool(seq)
    ih_off = IH_OFF_DEFAULT
    if train_opts_record is not None and "ih_off" in train_opts_record:
        ih_off = str(train_opts_record["ih_off"])
    if model == 'borst':
        base = _schema_from_opts(model, model_backend, schema, train_opts_record)
        sch = borst_schema(model_backend, base, ih_off)
    elif schema is not None:
        sch = list(schema)
    else:
        sch = _schema_from_opts(model, model_backend, None, train_opts_record)
    if train_opts_record is not None:
        train_opts_record["param_partitions"] = schema_partitions_record(
            sch, lambda seg: unit_names_for_segment(seg, model_backend),
        )
    sch = attach_param_carry(sch)
    session = TrainSession(
        backend=model_backend,
        model=model,
        schema=tuple(sch),
        targets=dict(packs),
        target_list=tuple(target_list),
        cost_weights=expand_cost_weight_dict(cost_weights),
        sequential=bool(seq),
        device=dev_ref,
        sim_dtype=sim_dtype,
        train_opts=train_opts_record,
    )
    cost_subpacks = _build_cost_subpacks(session)
    fused_borst = _build_fused_borst(session, cost_subpacks)
    return replace(session, cost_subpacks=cost_subpacks, fused_borst=fused_borst)


def open_session(
    opts: dict,
    model: str,
    *,
    schema: Optional[list] = None,
    model_backend: Optional[ModelBackend] = None,
) -> TrainSession:
    """Build a :class:`TrainSession` from canonical training opts."""
    backend_name = str(opts.get("backend", "network"))
    if backend_name != "network":
        raise ValueError(f"backend must be 'network', got {backend_name!r}")
    target_list = normalize_target_list(opts.get("target_list"))
    bad = [t for t in target_list if t not in VALID_TARGETS]
    if bad:
        raise ValueError(f"unknown target(s) {bad!r} (expected {'|'.join(CLI_TARGET_NAMES)})")
    dev = opts.get("dev") or active_device()
    sim_dtype = sim_dtype_from_fp32(bool(opts.get("fp32", False)))

    C = opts.get("network")
    syn_mode = normalize_syn_mode(opts.get("syn_mode", SYN_MODE_DEFAULT))
    if C is None:
        nj = opts.get("network_json")
        if not nj:
            raise ValueError("open_session(network) requires opts['network'] or network_json")
        C = load_network(
            nj, device=dev,
            exc_synweight=exc_synweight, inh_synweight=inh_synweight,
            dtype=sim_dtype, syn_mode=syn_mode,
        )
    if model_backend is None:
        model_backend = _network_backend_from_connectome(C, sim_dtype=sim_dtype)
    elif model_backend.network is not C:
        raise ValueError("model_backend.network must be opts['network']")
    ctx = _TrainBindCtx(
        model_backend=model_backend,
        dev=dev,
        sim_dtype=sim_dtype,
        cost_weights=opts.get("cost_weights"),
        spot_bright_stimulus_opts=opts.get("spot_bright_stimulus_opts"),
        spot_dark_stimulus_opts=opts.get("spot_dark_stimulus_opts"),
        moving_bar_bright_stimulus_opts=opts.get("moving_bar_bright_stimulus_opts"),
        moving_bar_dark_stimulus_opts=opts.get("moving_bar_dark_stimulus_opts"),
    )
    packs = {}
    pack_overrides = opts.get("pack_overrides") or {}
    resolved_spot_bright = resolved_spot_dark = None
    resolved_bar_bright = resolved_bar_dark = None
    for tname in target_list:
        pack, stim, _tag = NETWORK_TARGET_BUILDERS[tname](ctx, C)
        if tname in pack_overrides:
            pack = apply_pack_override(pack, pack_overrides[tname], model_backend)
        packs[tname] = pack
        if tname == "spot_bright":
            resolved_spot_bright = stim
        elif tname == "spot_dark":
            resolved_spot_dark = stim
        elif tname == "moving_bar_bright":
            resolved_bar_bright = stim
        elif tname == "moving_bar_dark":
            resolved_bar_dark = stim
    record = _train_opts_for_sidecar(
        opts, "network", target_list,
        resolved_spot_bright, resolved_spot_dark,
        resolved_bar_bright, resolved_bar_dark, False,
    )
    return _make_session(
        model_backend, model, target_list, packs,
        cost_weights=opts.get("cost_weights"),
        sequential=opts.get("sequential"),
        dev=dev,
        train_opts_record=record,
        schema=schema,
        sim_dtype=sim_dtype,
    )


def open_session_from_opts(opts: dict, model: str | None = None, **kwargs) -> TrainSession:
    """Restore a session from a saved ``train_opts.json`` dict."""
    opts = dict(opts)
    if model is None:
        model = opts.get("model")
        if not model:
            raise ValueError("train_opts requires model")
    opts["packs"] = None
    backend = str(opts.get("backend", "network"))
    if backend != "network":
        raise ValueError(f"backend must be 'network', got {backend!r}")
    nj = opts.get("network_json")
    if not nj:
        raise ValueError("train_opts requires network_json")
    if not opts.get("target_list"):
        raise ValueError("train_opts requires target_list")
    sim_dtype = sim_dtype_from_fp32(bool(opts.get("fp32", False)))
    syn_mode = normalize_syn_mode(opts.get("syn_mode", SYN_MODE_DEFAULT))
    mb = load_network_backend(
        nj, dev=opts.get("dev") or active_device(), sim_dtype=sim_dtype,
        syn_mode=syn_mode,
    )
    opts["network"] = mb.network
    opts["syn_mode"] = syn_mode
    kwargs.setdefault("model_backend", mb)
    return open_session({**opts, "backend": "network"}, model, **kwargs)


def open_session_from_outdir(
    outdir: str,
    model: str | None = None,
) -> TrainSession:
    """Load ``train_opts.json`` from a run folder and return a ready session."""
    opts_path = os.path.join(run_data_dir(os.path.abspath(outdir)), TRAIN_OPTS_FILE)
    if not os.path.isfile(opts_path):
        raise FileNotFoundError(f"missing {opts_path}")
    with open(opts_path) as f:
        opts = json.load(f)
    return open_session_from_opts(opts, model)
