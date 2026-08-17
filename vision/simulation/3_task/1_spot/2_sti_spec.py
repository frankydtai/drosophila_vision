# -*- coding: utf-8 -*-
"""Spot sti spec: timing, ``sti_mask``, and ``i_sti`` assembly.

The on/off ``sti_mask[t]`` is defined once (:func:`sti_mask`) and consumed by
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


def _merge_filter_ms(sti_opts: dict, sti_timing_key: str, ms) -> None:
    if ms is None:
        return
    if isinstance(ms, (int, float)):
        sti_opts[sti_timing_key] = {"v": float(ms), "ca": float(ms)}
        return
    if not isinstance(ms, dict):
        raise TypeError(
            f"{sti_timing_key} must be a float or {{v, ca}} dict, got {type(ms)!r}"
        )
    merged = {
        filter: float(ms_at_filter)
        for filter, ms_at_filter in (sti_opts.get(sti_timing_key) or {}).items()
    }
    merged.update(
        (filter, float(ms_at_filter)) for filter, ms_at_filter in ms.items()
    )
    sti_opts[sti_timing_key] = merged


def standardize_sti_timing(sti_opts: dict) -> dict:
    """In-place: per filter ``ms_response = max(ms_response, ms_sti)``."""
    ms_sti = sti_opts.get("ms_sti")
    ms_response = sti_opts.get("ms_response")
    if ms_sti is None or ms_response is None:
        return sti_opts
    if isinstance(ms_sti, dict) and isinstance(ms_response, dict):
        for filter in set(ms_sti) | set(ms_response):
            ms_response[filter] = max(float(ms_response[filter]), float(ms_sti[filter]))
    else:
        sti_opts["ms_response"] = max(float(ms_response), float(ms_sti))
    return sti_opts


def _float_ms(ms):
    if ms is None:
        return None
    if isinstance(ms, dict):
        return {
            filter: float(ms_at_filter) for filter, ms_at_filter in ms.items()
        }
    return float(ms)


def override_sti_timing(
    sti_opts: dict,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_sti=None,
    delta_ms=None,
    delta_ms_pre=None,
) -> dict:
    """Merge non-None timing into ``sti_opts``, standardize, drop derived ``t_onset``/``n_t``.

    Returns timing tokens whose values differ from the pre-merge snapshot (for
    plot / analyze filename suffixes).
    """
    before = {}
    for sti_timing_key in _STI_TIMING_KEYS:
        ms = sti_opts.get(sti_timing_key)
        before[sti_timing_key] = dict(ms) if isinstance(ms, dict) else ms
    for sti_timing_key, ms in (
        ("ms_pre", ms_pre),
        ("ms_post", ms_post),
        ("delta_ms", delta_ms),
        ("delta_ms_pre", delta_ms_pre),
        ("ms_response", ms_response),
        ("ms_sti", ms_sti),
    ):
        _merge_filter_ms(sti_opts, sti_timing_key, ms)
    standardize_sti_timing(sti_opts)
    sti_opts.pop("t_onset", None)
    sti_opts.pop("n_t", None)
    return {
        sti_timing_key: sti_opts.get(sti_timing_key)
        for sti_timing_key in _STI_TIMING_KEYS
        if _float_ms(before[sti_timing_key]) != _float_ms(sti_opts.get(sti_timing_key))
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
    """Inclusive last sti-on sample index (matches ``sti_mask``).

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


def sti_mask(t_onset, n_t, ms_sti=None, *, delta_ms: float) -> np.ndarray:
    """0/1 sti-on mask over ``n_t`` samples.

    ``ms_sti`` omitted → continue-on step (``sti_mask[t_onset:] = 1``). With a
    value the sti is on for inclusive ``[t_onset, t_sti_end(...)]`` (slice
    ``[t_onset, t_onset + round(ms_sti/delta_ms))``) and returns to baseline
    afterward; ``n_t`` is unchanged.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    mask = np.zeros(n_t)
    if ms_sti is None:
        mask[t_onset:] = 1.0
    else:
        n_t_on = max(1, t_from_ms(ms_sti, delta_ms=delta_ms))
        mask[t_onset:min(n_t, t_onset + n_t_on)] = 1.0
    return mask


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
    i_sti: float,
    sim_dtype,
    device,
):
    """Baseline ``i_sti`` + center bake + radius contribs for ``a_sti_radius``.

    Returns ``(i_sti, i_sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs)`` where
    center radius=0 is baked into ``i_sti`` at scale 1, and radius contribs
    compose as ``i_sti += a_sti_radius[radius] * i_sti_pulse`` on
    ``(sti_bs, sti_nodes)``. ``i_sti_pulse`` is
    ``(i_sti - i_baseline) * sti_mask(...)``. ``a_sti_radius_idxs`` indexes
    ``a_sti_radii`` / ``a_sti_radius`` (center radius=0 not in that axis). Empty
    ``a_sti_radii`` → center-only drive. Does not modify gt construction.
    """
    radii = tuple(int(radius) for radius in a_sti_radii)
    if any(radius == 0 for radius in radii):
        raise ValueError("a_sti_radii must omit center radius=0 (baked into i_sti @1)")
    radius_idx = dict(zip(radii, range(len(radii))))
    sti_bs: list[int] = []
    sti_nodes: list[int] = []
    a_sti_radius_idxs: list[int] = []
    center_nodes: list[tuple[int, int]] = []
    for b, spot_b in enumerate(spot_bs):
        for sti_hex_u, sti_hex_v in spot_b.sti_uv:
            for node in connectome.sti_nodes_at_uv(int(sti_hex_u), int(sti_hex_v)):
                center_nodes.append((int(b), int(node)))
            for radius in radii:
                a_sti_radius_idx = radius_idx[radius]
                for du, dv in build_hex.shell_hexes(radius):
                    for node in connectome.sti_nodes_at_uv(int(sti_hex_u) + int(du), int(sti_hex_v) + int(dv)):
                        sti_bs.append(int(b))
                        sti_nodes.append(int(node))
                        a_sti_radius_idxs.append(int(a_sti_radius_idx))
    mask = sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    n_b = len(spot_bs)
    network_sti_nodes = torch.as_tensor(connectome.sti_nodes, dtype=torch.long, device=device)
    i_sti_pulse = torch.as_tensor(
        (float(i_sti) - float(i_baseline)) * mask, dtype=sim_dtype, device=device,
    )
    i_sti = torch.zeros((n_b, n_t, connectome.n_node), dtype=sim_dtype, device=device)
    if len(network_sti_nodes):
        i_sti[:, :, network_sti_nodes] = float(i_baseline)
    for b, node in center_nodes:
        i_sti[b, :, node] = i_sti[b, :, node] + i_sti_pulse
    sti_bs = torch.tensor(sti_bs, dtype=torch.long, device=device)
    sti_nodes = torch.tensor(sti_nodes, dtype=torch.long, device=device)
    a_sti_radius_idxs = torch.tensor(a_sti_radius_idxs, dtype=torch.long, device=device)
    return i_sti, i_sti_pulse, sti_bs, sti_nodes, a_sti_radius_idxs
