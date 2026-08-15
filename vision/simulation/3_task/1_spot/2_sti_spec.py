# -*- coding: utf-8 -*-
"""Spot paradigm sti spec: timing, drive waveform, and ``i_sti`` assembly.

The sti drive waveform ``u[t]`` is defined once (``sti_waveform``) and
consumed by both the network ``i_sti`` and the ir component in
:mod:`task.spot.gt` (and ``i_sti`` via :mod:`task.spot.pack`), so sti-on
duration has a single source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from neuron.param import t_from_ms
from task.spot.sti_geo import SpotBatch, members_by_radius

_STI_TIMING_TOKENS = (
    "ms_pre", "ms_response", "ms_post", "ms_sti", "delta_ms", "delta_ms_pre",
)


@dataclass(frozen=True)
class StiTiming:
    ms_pre: float
    ms_response: float
    ms_post: float
    ms_sti: Optional[float]
    delta_ms: float
    delta_ms_pre: float
    t_onset: int
    n_t: int
    n_t_gt: int


def _timing_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        timing_toks = set(a) | set(b)
        return all(_timing_equal(a.get(k), b.get(k)) for k in timing_toks)
    return float(a) == float(b)


def _merge_filter_branch_ms(so: dict, timing_tok: str, val) -> None:
    if val is None:
        return
    if isinstance(val, dict):
        cur = so.get(timing_tok)
        if cur is None:
            so[timing_tok] = {k: float(v) for k, v in val.items()}
        else:
            merged = {k: float(v) for k, v in cur.items()}
            merged.update({k: float(v) for k, v in val.items()})
            so[timing_tok] = merged
    elif isinstance(val, (int, float)):
        so[timing_tok] = {"v": float(val), "ca": float(val)}
    else:
        raise TypeError(
            f"{timing_tok} must be a float or {{v, ca}} dict, got {type(val)!r}"
        )


def normalize_sti_timing(so: dict) -> dict:
    """In-place: per branch ``ms_response = max(ms_response, ms_sti)``."""
    ms_sti = so.get("ms_sti")
    resp = so.get("ms_response")
    if ms_sti is None or resp is None:
        return so
    if isinstance(ms_sti, dict) and isinstance(resp, dict):
        for branch in set(ms_sti) | set(resp):
            sti_val = float(ms_sti[branch])
            resp_val = float(resp[branch])
            if resp_val < sti_val:
                resp[branch] = sti_val
    else:
        if float(resp) < float(ms_sti):
            so["ms_response"] = float(ms_sti)
    return so


def override_sti_timing(
    so: dict,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
) -> dict:
    """Merge non-None timing into ``so``, normalize, drop derived ``t_onset``/``n_t``.

    Returns timing tokens whose values differ from the pre-merge snapshot (for
    plot / analyze filename suffixes).
    """
    before = {k: so.get(k) for k in _STI_TIMING_TOKENS}
    for timing_tok, val in (
        ("ms_pre", ms_pre),
        ("ms_post", ms_post),
    ):
        if val is not None:
            _merge_filter_branch_ms(so, timing_tok, val)
    _merge_filter_branch_ms(so, "delta_ms", delta_ms)
    _merge_filter_branch_ms(so, "delta_ms_pre", delta_ms_pre)
    _merge_filter_branch_ms(so, "ms_response", ms_response)
    _merge_filter_branch_ms(so, "ms_sti", ms_sti)
    normalize_sti_timing(so)
    so.pop("t_onset", None)
    so.pop("n_t", None)
    return {
        k: so.get(k)
        for k in _STI_TIMING_TOKENS
        if not _timing_equal(before.get(k), so.get(k))
    }


def sti_timing_t(
    *,
    ms_pre: float,
    ms_response: float,
    delta_ms: float,
    delta_ms_pre: float,
    ms_post: float = 0.0,
) -> tuple[int, int]:
    """Return ``(t_onset, n_t)`` from ms timing params.

    Pre uses ``delta_ms_pre``; response / post use ``delta_ms``. ``n_t`` is the
    forward length: pre + response + post. Cost / gt use only through
    response (``ms_post=0``); ``ms_post`` does not enter gt.
    """
    dt = float(delta_ms)
    dt_pre = float(delta_ms_pre)
    t_onset = t_from_ms(ms_pre, delta_ms=dt_pre)
    n_t = (
        t_onset
        + t_from_ms(ms_response, delta_ms=dt)
        + t_from_ms(ms_post, delta_ms=dt)
        + 1
    )
    return t_onset, n_t


def resolve_sti_timing(opts) -> StiTiming:
    if opts.get("ms_pre") is None or opts.get("ms_response") is None:
        raise ValueError(
            "spot sti opts require ms_pre and ms_response "
            "(pass via CLI --sti-timing ms_pre=… ms_response=…)"
        )
    if opts.get("delta_ms") is None:
        raise ValueError("spot sti opts require delta_ms")
    if opts.get("delta_ms_pre") is None:
        raise ValueError("spot sti opts require delta_ms_pre")
    ms_pre = float(opts["ms_pre"])
    ms_response = float(opts["ms_response"])
    delta_ms = float(opts["delta_ms"])
    delta_ms_pre = float(opts["delta_ms_pre"])
    ms_post = float(opts.get("ms_post", 0.0))
    raw_ms_sti = opts.get("ms_sti")
    ms_sti = None
    if raw_ms_sti is not None:
        ms_sti = float(raw_ms_sti)
    t_onset, n_t = sti_timing_t(
        ms_pre=ms_pre,
        ms_response=ms_response,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_post=ms_post,
    )
    n_t_gt = sti_timing_t(
        ms_pre=ms_pre,
        ms_response=ms_response,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_post=0.0,
    )[1]
    return StiTiming(
        ms_pre=ms_pre,
        ms_response=ms_response,
        ms_post=ms_post,
        ms_sti=ms_sti,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        t_onset=int(t_onset),
        n_t=int(n_t),
        n_t_gt=int(n_t_gt),
    )


def resolve_sti_timing_t(opts) -> tuple[int, int]:
    """``(t_onset, n_t)`` from sti opts timing + piecewise ``delta_ms*``.

    Optional ``ms_post`` (default 0) extends forward only.
    """
    timing = resolve_sti_timing(opts)
    return timing.t_onset, timing.n_t


def resolve_sti_gt_n_t(opts) -> int:
    """Cost ``n_t`` from opts (ignores ``ms_post``)."""
    return resolve_sti_timing(opts).n_t_gt


def t_sti_end(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> int:
    """Inclusive last sti-on sample index (matches ``sti_waveform``).

    On samples are ``[t_onset, t_sti_end]``. With ``ms_sti``, that is
    ``t_onset + max(1, round(ms_sti/delta_ms)) - 1`` (clamped to ``n_t - 1``).
    ``ms_sti`` omitted → continue-on through the last sample (``n_t - 1``).
    """
    t0 = int(t_onset)
    mt = int(n_t)
    if mt <= 0:
        raise ValueError(f"n_t must be positive, got {n_t}")
    if ms_sti is None:
        return mt - 1
    width = max(1, t_from_ms(float(ms_sti), delta_ms=float(delta_ms)))
    return min(mt - 1, t0 + width - 1)


def sti_waveform(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> np.ndarray:
    """Normalized 0/1 sti drive ``u[t]`` over ``n_t`` samples.

    ``ms_sti`` omitted -> continue-on step (``u[t_onset:] = 1``). With a value the
    sti is on for inclusive ``[t_onset, t_sti_end(...)]`` (slice
    ``[t_onset, t_onset + round(ms_sti/delta_ms))``) and returns to baseline
    afterward; ``n_t`` is unchanged.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    u = np.zeros(n_t)
    if ms_sti is None:
        u[t_onset:] = 1.0
    else:
        width = max(1, t_from_ms(ms_sti, delta_ms=delta_ms))
        u[t_onset:min(n_t, t_onset + width)] = 1.0
    return u


def build_spot_a_sti_radius_drive(
    connectome,
    batches: Sequence[SpotBatch],
    *,
    a_sti_radii,
    t_onset: int,
    n_t: int,
    ms_sti,
    delta_ms: float,
    i_baseline: float,
    i_peak: float,
    sim_dtype,
    device,
):
    """Baseline ``i_sti`` + center bake + radius contribs for ``a_sti_radius``.

    Returns ``(i_sti, sti_wave, sti_batches, sti_nodes, a_sti_radius_indices)`` where center
    r=0 is baked into ``i_sti`` at scale 1, and radius contribs compose as
    ``i += a_sti_radius[r] * sti_wave`` on ``(sti_batches, sti_nodes)``. ``a_sti_radius_indices``
    indexes ``a_sti_radii`` / ``a_sti_radius`` (no center slot). Empty
    ``a_sti_radii`` → center-only drive. Does not modify gt construction.
    """
    radii = tuple(int(r) for r in a_sti_radii)
    if any(r == 0 for r in radii):
        raise ValueError("a_sti_radii must omit center r=0 (baked into i_sti @1)")
    by_radius = members_by_radius(radii) if radii else {}
    idx_from_radius = {r: i for i, r in enumerate(radii)}
    batch_l: list[int] = []
    node_l: list[int] = []
    r_l: list[int] = []
    center_nodes: list[tuple[int, int]] = []
    for b, batch in enumerate(batches):
        for sti_hex_u, sti_hex_v in batch.sti_uv:
            for nid in connectome.sti_nodes_at(int(sti_hex_u), int(sti_hex_v)):
                center_nodes.append((int(b), int(nid)))
            for radius, members in by_radius.items():
                ri = idx_from_radius[radius]
                for du, dv in members:
                    for nid in connectome.sti_nodes_at(int(sti_hex_u) + int(du), int(sti_hex_v) + int(dv)):
                        batch_l.append(int(b))
                        node_l.append(int(nid))
                        r_l.append(int(ri))
    u = sti_waveform(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    n_batch = len(batches)
    sti_idx = torch.as_tensor(connectome.sti_node_indices, dtype=torch.long, device=device)
    i_sti = torch.zeros((n_batch, n_t, connectome.n_nodes), dtype=sim_dtype, device=device)
    if len(sti_idx):
        i_sti[:, :, sti_idx] = float(i_baseline)
    sti_wave = torch.as_tensor(
        (float(i_peak) - float(i_baseline)) * u, dtype=sim_dtype, device=device,
    )
    for b, nid in center_nodes:
        i_sti[b, :, nid] = i_sti[b, :, nid] + sti_wave
    sti_batches = torch.tensor(batch_l, dtype=torch.long, device=device)
    sti_nodes = torch.tensor(node_l, dtype=torch.long, device=device)
    a_sti_radius_indices = torch.tensor(r_l, dtype=torch.long, device=device)
    return i_sti, sti_wave, sti_batches, sti_nodes, a_sti_radius_indices
