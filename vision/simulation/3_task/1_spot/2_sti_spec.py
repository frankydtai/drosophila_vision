# -*- coding: utf-8 -*-
"""Spot sti spec: timing, sti pulse, and ``i_sti`` assembly.

The sti pulse ``pulse[t]`` is defined once (``sti_pulse``) and consumed by
both the network ``i_sti`` and the ir component in :mod:`task.spot.gt`
(and ``i_sti`` via :mod:`task.spot.pack`), so sti-on duration has a single
source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex
from neuron.param import t_from_ms
from task.spot.sti_geo import SpotB

_STI_TIMING_KEYS = (
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
        sti_timing_keys = set(a) | set(b)
        return all(_timing_equal(a.get(k), b.get(k)) for k in sti_timing_keys)
    return float(a) == float(b)


def _merge_filter_branch_ms(so: dict, sti_timing_key: str, val) -> None:
    if val is None:
        return
    if isinstance(val, dict):
        cur = so.get(sti_timing_key)
        if cur is None:
            so[sti_timing_key] = {k: float(v) for k, v in val.items()}
        else:
            merged = {k: float(v) for k, v in cur.items()}
            merged.update({k: float(v) for k, v in val.items()})
            so[sti_timing_key] = merged
    elif isinstance(val, (int, float)):
        so[sti_timing_key] = {"v": float(val), "ca": float(val)}
    else:
        raise TypeError(
            f"{sti_timing_key} must be a float or {{v, ca}} dict, got {type(val)!r}"
        )


def standardize_sti_timing(so: dict) -> dict:
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
    """Merge non-None timing into ``so``, standardize, drop derived ``t_onset``/``n_t``.

    Returns timing tokens whose values differ from the pre-merge snapshot (for
    plot / analyze filename suffixes).
    """
    before = {k: so.get(k) for k in _STI_TIMING_KEYS}
    for sti_timing_key, val in (
        ("ms_pre", ms_pre),
        ("ms_post", ms_post),
    ):
        if val is not None:
            _merge_filter_branch_ms(so, sti_timing_key, val)
    _merge_filter_branch_ms(so, "delta_ms", delta_ms)
    _merge_filter_branch_ms(so, "delta_ms_pre", delta_ms_pre)
    _merge_filter_branch_ms(so, "ms_response", ms_response)
    _merge_filter_branch_ms(so, "ms_sti", ms_sti)
    standardize_sti_timing(so)
    so.pop("t_onset", None)
    so.pop("n_t", None)
    return {
        k: so.get(k)
        for k in _STI_TIMING_KEYS
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


def t_sti_end(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> int:
    """Inclusive last sti-on sample index (matches ``sti_pulse``).

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
    w = max(1, t_from_ms(float(ms_sti), delta_ms=float(delta_ms)))
    return min(mt - 1, t0 + w - 1)


def sti_pulse(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> np.ndarray:
    """Normalized 0/1 sti pulse over ``n_t`` samples.

    ``ms_sti`` omitted -> continue-on step (``pulse[t_onset:] = 1``). With a
    value the sti is on for inclusive ``[t_onset, t_sti_end(...)]`` (slice
    ``[t_onset, t_onset + round(ms_sti/delta_ms))``) and returns to baseline
    afterward; ``n_t`` is unchanged.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    pulse = np.zeros(n_t)
    if ms_sti is None:
        pulse[t_onset:] = 1.0
    else:
        w = max(1, t_from_ms(ms_sti, delta_ms=delta_ms))
        pulse[t_onset:min(n_t, t_onset + w)] = 1.0
    return pulse


def build_spot_a_sti_radius_drive(
    connectome,
    spot_bs: Sequence[SpotB],
    *,
    a_sti_radii,
    t_onset: int,
    n_t: int,
    ms_sti,
    delta_ms: float,
    i_baseline: float,
    i_spot: float,
    sim_dtype,
    device,
):
    """Baseline ``i_sti`` + center bake + radius contribs for ``a_sti_radius``.

    Returns ``(i_sti, sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs)`` where
    center radius=0 is baked into ``i_sti`` at scale 1, and radius contribs
    compose as ``i += a_sti_radius[radius] * sti_pulse`` on
    ``(sti_bs, sti_nodes)``. ``sti_pulse`` is
    ``(i_spot - i_baseline) * sti_pulse(...)``. ``a_sti_radius_idxs`` indexes
    ``a_sti_radii`` / ``a_sti_radius`` (center radius=0 not in that axis). Empty
    ``a_sti_radii`` → center-only drive. Does not modify gt construction.
    """
    radii = tuple(int(radius) for radius in a_sti_radii)
    if any(radius == 0 for radius in radii):
        raise ValueError("a_sti_radii must omit center radius=0 (baked into i_sti @1)")
    radius_idx = dict(zip(radii, range(len(radii))))
    sti_b_vals: list[int] = []
    node_l: list[int] = []
    r_l: list[int] = []
    center_nodes: list[tuple[int, int]] = []
    for b, spot_b in enumerate(spot_bs):
        for sti_hex_u, sti_hex_v in spot_b.sti_uv:
            for node in connectome.sti_nodes_at_uv(int(sti_hex_u), int(sti_hex_v)):
                center_nodes.append((int(b), int(node)))
            for radius in radii:
                ri = radius_idx[radius]
                for du, dv in build_hex.shell_hexes(radius):
                    for node in connectome.sti_nodes_at_uv(int(sti_hex_u) + int(du), int(sti_hex_v) + int(dv)):
                        sti_b_vals.append(int(b))
                        node_l.append(int(node))
                        r_l.append(int(ri))
    pulse = sti_pulse(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    n_b = len(spot_bs)
    sti_nodes = torch.as_tensor(connectome.sti_nodes, dtype=torch.long, device=device)
    i_sti = torch.zeros((n_b, n_t, connectome.n_nodes), dtype=sim_dtype, device=device)
    if len(sti_nodes):
        i_sti[:, :, sti_nodes] = float(i_baseline)
    # Amplitude-scaled pulse for center bake and a_sti_radius inject (Pack.sti_pulse).
    scaled = torch.as_tensor(
        (float(i_spot) - float(i_baseline)) * pulse, dtype=sim_dtype, device=device,
    )
    for b, node in center_nodes:
        i_sti[b, :, node] = i_sti[b, :, node] + scaled
    sti_bs = torch.tensor(sti_b_vals, dtype=torch.long, device=device)
    sti_nodes = torch.tensor(node_l, dtype=torch.long, device=device)
    a_sti_radius_idxs = torch.tensor(r_l, dtype=torch.long, device=device)
    return i_sti, scaled, sti_bs, sti_nodes, a_sti_radius_idxs
