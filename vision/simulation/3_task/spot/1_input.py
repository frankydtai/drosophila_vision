# -*- coding: utf-8 -*-
"""Spot paradigm INPUT: connectome spot geometry + PR drive waveform.

Geometry (centers, sub-spot shifts, Euclidean radii) is split out of the old
``network.spot_target`` Section A. The PR drive waveform ``u[t]`` is defined
here once (``spot_input_waveform``) and consumed by both the network ``i_sti`` and
the ImpR gt in :mod:`task.spot.gt` (and ``i_sti`` via :mod:`task.spot.readout`),
so spot-on duration has a single source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex

from neuron.params import t_from_ms

_SPOT_RADIUS_HALF_STEP_TOL = 1e-9

_SPOT_TIMING_KEYS = (
    "ms_pre", "ms_response", "ms_post", "ms_spot", "delta_ms", "delta_ms_pre",
)


def _timing_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return float(a) == float(b)


def normalize_spot_timing(so: dict) -> dict:
    """In-place: ``ms_response = max(ms_response, ms_spot)`` when both set."""
    ms_spot = so.get("ms_spot")
    resp = so.get("ms_response")
    if ms_spot is not None and resp is not None and float(resp) < float(ms_spot):
        so["ms_response"] = float(ms_spot)
    return so


def apply_spot_timing_overrides(
    so: dict,
    *,
    ms_pre=None,
    ms_response=None,
    ms_post=None,
    ms_spot=None,
    delta_ms=None,
    delta_ms_pre=None,
) -> dict:
    """Merge non-None timing into ``so``, normalize, drop derived ``t_onset``/``n_t``.

    Returns timing keys whose values differ from the pre-merge snapshot (for
    plot / analyze filename suffixes).
    """
    before = {k: so.get(k) for k in _SPOT_TIMING_KEYS}
    for key, val in (
        ("ms_pre", ms_pre),
        ("ms_response", ms_response),
        ("ms_post", ms_post),
        ("ms_spot", ms_spot),
        ("delta_ms", delta_ms),
        ("delta_ms_pre", delta_ms_pre),
    ):
        if val is not None:
            so[key] = float(val)
    normalize_spot_timing(so)
    so.pop("t_onset", None)
    so.pop("n_t", None)
    return {
        k: so.get(k)
        for k in _SPOT_TIMING_KEYS
        if not _timing_equal(before.get(k), so.get(k))
    }


def spot_timing_t(
    *,
    ms_pre: float,
    ms_response: float,
    delta_ms: float,
    delta_ms_pre: float,
    ms_post: float = 0.0,
) -> tuple[int, int]:
    """Return ``(t_onset, n_t)`` from ms timing params.

    Pre uses ``delta_ms_pre``; response / post use ``delta_ms``. ``n_t`` is the
    forward length: pre + response + post. Cost / ImpR gt use only through
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


def _require_spot_timing_opts(opts):
    if opts.get("ms_pre") is None or opts.get("ms_response") is None:
        raise ValueError(
            "spot stimulus opts require ms_pre and ms_response "
            "(pass via CLI --ms-pre / --ms-response)"
        )
    if opts.get("delta_ms") is None:
        raise ValueError("spot stimulus opts require delta_ms")
    if opts.get("delta_ms_pre") is None:
        raise ValueError("spot stimulus opts require delta_ms_pre")
    return (
        float(opts["ms_pre"]),
        float(opts["ms_response"]),
        float(opts["delta_ms"]),
        float(opts["delta_ms_pre"]),
        float(opts.get("ms_post", 0.0)),
    )


def spot_timing_t_from_opts(opts) -> tuple[int, int]:
    """``(t_onset, n_t)`` from stimulus opts timing + piecewise ``delta_ms*``.

    Optional ``ms_post`` (default 0) extends forward only.
    """
    ms_pre, ms_response, delta_ms, delta_ms_pre, ms_post = _require_spot_timing_opts(opts)
    return spot_timing_t(
        ms_pre=ms_pre,
        ms_response=ms_response,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_post=ms_post,
    )


def spot_gt_n_t_from_opts(opts) -> int:
    """ImpR / cost ``n_t`` from opts (ignores ``ms_post``)."""
    ms_pre, ms_response, delta_ms, delta_ms_pre, _ = _require_spot_timing_opts(opts)
    return spot_timing_t(
        ms_pre=ms_pre,
        ms_response=ms_response,
        delta_ms=delta_ms,
        delta_ms_pre=delta_ms_pre,
        ms_post=0.0,
    )[1]


def spot_t_spot_end(t_onset, n_t, ms_spot=None, *, delta_ms: float) -> int:
    """Inclusive last stimulus-on sample index (matches ``spot_input_waveform``).

    On samples are ``[t_onset, t_spot_end]``. With ``ms_spot``, that is
    ``t_onset + max(1, round(ms_spot/delta_ms)) - 1`` (clamped to ``n_t - 1``).
    ``ms_spot`` omitted → continue-on through the last sample (``n_t - 1``).
    """
    t0 = int(t_onset)
    mt = int(n_t)
    if mt <= 0:
        raise ValueError(f"n_t must be positive, got {n_t}")
    if ms_spot is None:
        return mt - 1
    width = max(1, t_from_ms(float(ms_spot), delta_ms=float(delta_ms)))
    return min(mt - 1, t0 + width - 1)


def spot_input_waveform(t_onset, n_t, ms_spot=None, *, delta_ms: float) -> np.ndarray:
    """Normalized 0/1 photoreceptor drive ``u[t]`` over ``n_t`` samples.

    ``ms_spot`` omitted -> continue-on step (``u[t_onset:] = 1``). With a value the
    stimulus is on for inclusive ``[t_onset, spot_t_spot_end(...)]`` (slice
    ``[t_onset, t_onset + round(ms_spot/delta_ms))``) and returns to baseline
    afterward; ``n_t`` is unchanged.
    """
    t_onset = int(t_onset)
    n_t = int(n_t)
    u = np.zeros(n_t)
    if ms_spot is None:
        u[t_onset:] = 1.0
    else:
        width = max(1, t_from_ms(ms_spot, delta_ms=delta_ms))
        u[t_onset:min(n_t, t_onset + width)] = 1.0
    return u


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in hex nodes) between two axial cells."""
    return math.sqrt(du * du + du * dv + dv * dv)


def members_by_euclid_radius(radii) -> dict[float, list[tuple[int, int]]]:
    """Map each Euclidean radius to stim-centered axial ``(du, dv)`` members."""
    radii_set = {round(float(radius), 6) for radius in radii}
    max_shell = int(math.ceil(max(radii_set)))
    by_radius: dict[float, list[tuple[int, int]]] = {radius: [] for radius in radii_set}
    for du, dv in build_hex.members_in_radius(max_shell):
        radius = round(euclid_hex_dist(du, dv), 6)
        if radius in radii_set:
            by_radius[radius].append((int(du), int(dv)))
    missing = [radius for radius in radii_set if not by_radius[radius]]
    if missing:
        raise ValueError(f"no hex members for spot cost radii {missing}")
    return by_radius


def spot_radius_half_steps(spot_radius) -> int:
    """``spot_radius = 0.5 * m`` for non-negative integer ``m``; return ``m``."""
    value = float(spot_radius)
    if value < 0:
        raise ValueError(f"spot_radius must be >= 0, got {spot_radius!r}")
    half_steps = value * 2.0
    m = round(half_steps)
    if abs(half_steps - m) > _SPOT_RADIUS_HALF_STEP_TOL:
        raise ValueError(
            f"spot_radius must be a non-negative 0.5 multiple, got {spot_radius!r}",
        )
    return int(m)


def spot_radius_dist(spot_radius) -> int:
    """Axial center spacing: ``2*spot_radius + 1`` (``spot_radius`` in 0.5 steps)."""
    return spot_radius_half_steps(spot_radius) + 1


def spot_radius_folds_r2_into_r1(spot_radius) -> bool:
    """True when ``spot_radius == 1`` (``spot_radius_half_steps == 2``).

    Fold semantics live in :func:`task.spot.gt._spot_readout_a_radius`: r=1 gt
    ``a_radius`` is ``RecF(1)+RecF(2)`` and r=2 ``a_radius`` is 0. Non-center drive
    scales use ``a_sti_radius`` gated by cost-radius weight (weight==0 → force
    0). Center r=0 remains baked at scale 1.
    """
    return spot_radius_half_steps(spot_radius) == 2


def _spot_center_angle(u: int, v: int) -> float:
    """Degree-space angle of (u, v), for a stable angular tie-break ordering."""
    x_deg, y_deg = build_hex.uv_to_xy_deg(u, v)
    return float(np.arctan2(float(y_deg), float(x_deg)))


def spot_centers(
    connectome_radius: int = build_hex.DEFAULT_RADIUS,
    *,
    spot_radius: float,
    fully_inside: bool,
) -> list:
    """Axial centers of densest packing of radius-``floor(spot_radius)`` hexes."""
    m = spot_radius_half_steps(spot_radius)
    k = m // 2
    if m % 2 == 1:
        e = (m + 1) // 2
        a1, b1 = e, e
    else:
        a1, b1 = m + 1, -k
    a2, b2 = -b1, a1 + b1  # 60° CCW about origin
    members = build_hex.members_in_radius((m + 1) // 2)
    span = int(2 * (connectome_radius // max(k, 1) + 2))
    centers: list = []
    for lm in range(-span, span + 1):
        for ln in range(-span, span + 1):
            cu = lm * a1 + ln * a2
            cv = lm * b1 + ln * b2
            if build_hex.hex_radius(cu, cv) > connectome_radius:
                continue
            if fully_inside and any(
                build_hex.hex_radius(cu + du, cv + dv) > connectome_radius
                for du, dv in members
            ):
                continue
            centers.append((cu, cv))
    centers.sort(
        key=lambda c: (build_hex.hex_radius(*c), _spot_center_angle(*c)),
    )
    return centers


@dataclass
class Spot:
    """Spot centers and sub-spot shifts over a loaded connectome."""

    centers: list[tuple[int, int]]
    shifts: list[tuple[int, int]]
    spot_radius: float


@dataclass(frozen=True)
class SpotBatch:
    """One simultaneous spot stimulus: all ``stim_uv`` hexes step in one batch."""

    shift: tuple[int, int]
    stim_uv: tuple[tuple[int, int], ...]


def spot_stimulus_batches(spot: Spot) -> list[SpotBatch]:
    """One batch per shift; each batch steps all spot centers (+ shift) together."""
    return [
        SpotBatch(
            shift=(int(du), int(dv)),
            stim_uv=tuple(
                (int(cu + du), int(cv + dv)) for cu, cv in spot.centers
            ),
        )
        for du, dv in spot.shifts
    ]


def _connectome_radius(C, spot_radius: float) -> int:
    """Hex-disc radius of connectome ``C``."""
    meta_radius = int(C.meta.get("extent", -1))
    if meta_radius >= 0:
        return meta_radius
    positioned = C.column_id >= 0
    radii = [
        build_hex.hex_radius(int(hex_u), int(hex_v))
        for hex_u, hex_v in zip(C.u[positioned], C.v[positioned])
    ]
    return max(radii) if radii else int(spot_radius)


def build_spot(
    C,
    *,
    spot_radius: float,
    multi_spot: bool,
    fully_inside: bool,
) -> Spot:
    """Build a :class:`Spot` for connectome ``C``."""
    spot_radius_half_steps(spot_radius)
    connectome_radius = _connectome_radius(C, spot_radius)
    shifts = build_hex.members_in_radius(1)
    if not multi_spot:
        centers = [(0, 0)]
    else:
        centers = [
            (int(cu), int(cv))
            for cu, cv in spot_centers(
                connectome_radius=connectome_radius,
                spot_radius=spot_radius,
                fully_inside=fully_inside,
            )
        ]
    return Spot(centers, shifts, spot_radius)


def spot_from_opts(
    C,
    *,
    spot_radius: float | None = None,
    shift_radius: int | None = None,
    multi_spot: bool | None = None,
    fully_inside: bool | None = None,
    stimulus_opts: dict | None = None,
) -> Spot:
    """Build :class:`Spot` with configurable sub-spot shift radius."""
    if stimulus_opts is not None:
        spot_radius = float(stimulus_opts["spot_radius"])
        shift_radius = int(stimulus_opts["shift_radius"])
        multi_spot = bool(stimulus_opts["multi_spot"])
        fully_inside = bool(stimulus_opts["fully_inside"])
    if (
        spot_radius is None
        or shift_radius is None
        or multi_spot is None
        or fully_inside is None
    ):
        raise TypeError(
            "spot_from_opts requires spot_radius, shift_radius, multi_spot, and "
            "fully_inside (or stimulus_opts containing them)"
        )
    spot = build_spot(
        C, spot_radius=spot_radius, multi_spot=multi_spot, fully_inside=fully_inside,
    )
    spot.shifts = [
        (int(du), int(dv))
        for du, dv in build_hex.members_in_radius(int(shift_radius))
    ]
    return spot


def build_spot_a_sti_radius_drive(
    C,
    batches: Sequence[SpotBatch],
    *,
    sti_radii,
    t_onset: int,
    n_t: int,
    ms_spot,
    delta_ms: float,
    i_baseline: float,
    i_peak: float,
    sim_dtype,
    device,
):
    """Baseline ``i_sti`` + center bake + radius contribs for ``a_sti_radius``.

    Returns ``(i_sti, sti_wave, sti_batch, sti_node, sti_radius)`` where center
    r=0 is baked into ``i_sti`` at scale 1, and radius contribs compose as
    ``i += a_sti_radius[r] * sti_wave`` on ``(sti_batch, sti_node)``. ``sti_radius``
    indexes ``sti_radii`` / ``a_sti_radius`` (no center slot). Empty
    ``sti_radii`` → center-only drive. Does not modify gt construction.
    """
    radii = tuple(round(float(r), 6) for r in sti_radii)
    if any(r == 0.0 for r in radii):
        raise ValueError("sti_radii must omit center r=0 (baked into i_sti @1)")
    by_radius = members_by_euclid_radius(radii) if radii else {}
    radius_to_i = {r: i for i, r in enumerate(radii)}
    batch_l: list[int] = []
    node_l: list[int] = []
    r_l: list[int] = []
    center_nodes: list[tuple[int, int]] = []
    for b, batch in enumerate(batches):
        for stim_hex_u, stim_hex_v in batch.stim_uv:
            for nid in C.input_nodes_at(int(stim_hex_u), int(stim_hex_v)):
                center_nodes.append((int(b), int(nid)))
            for radius_key, members in by_radius.items():
                ri = radius_to_i[radius_key]
                for du, dv in members:
                    for nid in C.input_nodes_at(int(stim_hex_u) + int(du), int(stim_hex_v) + int(dv)):
                        batch_l.append(int(b))
                        node_l.append(int(nid))
                        r_l.append(int(ri))
    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    n_batch = len(batches)
    pr_idx = torch.as_tensor(np.where(C.is_input)[0], dtype=torch.long, device=device)
    i_sti = torch.zeros((n_batch, n_t, C.n_nodes), dtype=sim_dtype, device=device)
    if len(pr_idx):
        i_sti[:, :, pr_idx] = float(i_baseline)
    sti_wave = torch.as_tensor(
        (float(i_peak) - float(i_baseline)) * u, dtype=sim_dtype, device=device,
    )
    for b, nid in center_nodes:
        i_sti[b, :, nid] = i_sti[b, :, nid] + sti_wave
    sti_batch = torch.tensor(batch_l, dtype=torch.long, device=device)
    sti_node = torch.tensor(node_l, dtype=torch.long, device=device)
    sti_radius = torch.tensor(r_l, dtype=torch.long, device=device)
    return i_sti, sti_wave, sti_batch, sti_node, sti_radius
