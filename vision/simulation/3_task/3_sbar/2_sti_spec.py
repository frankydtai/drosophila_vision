# -*- coding: utf-8 -*-
"""Static-bar sti spec: conditions, timing, contrast, and hex currents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from neuron.borst import t_from_ms
from task.sbar.sti_geo import (
    i_sti_nodes_from_hexes,
    sbar_line_hex_mask,
    sbar_line_mids,
    sbar_shift_mids,
    sti_hexes,
)
from task.spread.sti_spec import sti_mask

__all__ = (
    "SbarSpec",
    "SbarSti",
    "build_sbar_a_sti_mid_drive",
    "build_i_sti_hex",
    "build_sbar_signals",
    "gruntman_sbar_specs",
    "sbar_n_t",
)


def build_sbar_a_sti_mid_drive(
    connectome,
    specs: Sequence[SbarSpec],
    base_i_sti,
    *,
    a_sti_mids,
    bar_dist: int,
    multi_bar: bool,
    t_onset: int,
    n_t: int,
    ms_sti,
    delta_ms: float,
    i_baseline: float,
    i_sti: float,
):
    """Add trainable symmetric surround metadata to a fixed-line sbar drive.

    ``mid=0`` is already present at full amplitude in ``base_i_sti``.  Every
    configured positive distance is indexed by ``abs(axis_mid - bar_mid)``, so
    the two sides of a bar necessarily share one scalar.  Repeated entries are
    retained when surrounds from simultaneous bars overlap; ``index_add_`` in
    forward then sums them, matching simultaneous multi-spot stimulation.
    """
    mids = tuple(float(mid) for mid in a_sti_mids)
    if any(not np.isfinite(mid) or mid <= 0.0 for mid in mids):
        raise ValueError("a_sti_mids must contain finite positive distances; mid=0 is baked at 1")
    if len({round(mid, 9) for mid in mids}) != len(mids):
        raise ValueError("a_sti_mids must contain unique absolute distances")

    hexes = sti_hexes(connectome)
    sti_bs: list[int] = []
    sti_nodes: list[int] = []
    a_sti_mid_idxs: list[int] = []
    for b, spec in enumerate(specs):
        axis = "x" if spec.direction in ("right", "left") else "y"
        for bar_mid in sbar_line_mids(
            hexes, spec.direction, int(bar_dist), multi_bar=bool(multi_bar),
            shift_mid=spec.shift_mid,
        ):
            for sti_hex in hexes:
                axis_mid = float(sti_hex.x if axis == "x" else sti_hex.y)
                distance = abs(axis_mid - float(bar_mid))
                matches = [
                    mid_idx for mid_idx, mid in enumerate(mids)
                    if np.isclose(distance, mid, atol=1e-9, rtol=0.0)
                ]
                if not matches:
                    continue
                mid_idx = matches[0]
                for node in np.asarray(sti_hex.nodes).ravel():
                    sti_bs.append(int(b))
                    sti_nodes.append(int(node))
                    a_sti_mid_idxs.append(int(mid_idx))

    i_sti_pulse = (float(i_sti) - float(i_baseline)) * sti_mask(
        int(t_onset), int(n_t), ms_sti, delta_ms=float(delta_ms),
    )
    return (
        np.asarray(base_i_sti, dtype=np.float64).copy(),
        i_sti_pulse,
        np.asarray(sti_bs, dtype=np.int64),
        np.asarray(sti_nodes, dtype=np.int64),
        np.asarray(a_sti_mid_idxs, dtype=np.int64),
    )


@dataclass(frozen=True)
class SbarSpec:
    direction: str
    contrast: str
    shift_mid: float = 0.0

    @property
    def token(self) -> str:
        shift = (
            f"{int(self.shift_mid):+d}"
            if float(self.shift_mid).is_integer()
            else f"{float(self.shift_mid):+.1f}"
        )
        return f"{self.direction}_{self.contrast}_shift{shift}_w1"


def gruntman_sbar_specs(
    *,
    contrasts: Sequence[str],
    bar_directions: Sequence[str],
    shift_radius: int = 0,
) -> List[SbarSpec]:
    """Unique static-bar geometries over contrast and bar-normal shift.

    A static right-facing and left-facing bar are the same vertical line;
    likewise up/down are the same horizontal line.  Keep the first configured
    direction for each axis instead of running an identical stimulus twice.
    """
    specs: List[SbarSpec] = []
    seen = set()
    for direction in bar_directions:
        axis = "x" if direction in ("right", "left") else "y"
        for contrast in contrasts:
            for shift_mid in sbar_shift_mids(direction, int(shift_radius)):
                geometry = (str(contrast), axis, float(shift_mid))
                if geometry in seen:
                    continue
                seen.add(geometry)
                specs.append(SbarSpec(
                    direction=direction,
                    contrast=contrast,
                    shift_mid=shift_mid,
                ))
    return specs


def sbar_n_t(
    *,
    ms_pre: float,
    ms_response: float,
    ms_post: float = 0.0,
    ms_sti=None,
    delta_ms: float,
    delta_ms_pre: float,
) -> Tuple[int, int]:
    """Return ``(n_t, t_onset)`` from spread-style timing keys."""
    if ms_sti is not None:
        ms_response = max(float(ms_response), float(ms_sti))
    t_onset = int(t_from_ms(float(ms_pre), delta_ms=float(delta_ms_pre)))
    n_t = int(
        t_onset
        + t_from_ms(float(ms_response), delta_ms=float(delta_ms))
        + t_from_ms(float(ms_post), delta_ms=float(delta_ms))
        + 1
    )
    return n_t, t_onset


def build_i_sti_hex(
    hexes,
    specs: Sequence[SbarSpec],
    n_t: int,
    bar_dist: int,
    *,
    multi_bar: bool = True,
    t_onset: int,
    delta_ms: float,
    ms_sti,
    i_baseline: float,
    i_sti: float,
) -> np.ndarray:
    """Multi-b hex currents ``(B, T, n_hex)`` with fixed line bars and ``sti_mask`` timing."""
    n_b = len(specs)
    n_hex = len(hexes)
    if n_hex == 0 or n_b == 0:
        return np.zeros((n_b, n_t, n_hex), dtype=np.float64)

    mask = sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    i_sti_hex = np.full((n_b, n_t, n_hex), float(i_baseline), dtype=np.float64)

    by_geometry: dict[tuple[str, float], List[int]] = {}
    for b, spec in enumerate(specs):
        by_geometry.setdefault((spec.direction, spec.shift_mid), []).append(b)

    for (direction, shift_mid), bs in by_geometry.items():
        hex_mask = sbar_line_hex_mask(
            hexes, direction, bar_dist, multi_bar=multi_bar,
            shift_mid=shift_mid,
        )
        for b in bs:
            i_sti_hex[b] = float(i_baseline) + (
                hex_mask[np.newaxis, :] * (float(i_sti) - float(i_baseline)) * mask[:, np.newaxis]
            )
    return i_sti_hex


@dataclass
class SbarSti:
    i_sti: np.ndarray
    i_sti_hex: np.ndarray
    specs: List[SbarSpec]
    n_b: int
    n_t: int
    t_onset: int
    i_baseline: float
    bar_dist: int
    multi_bar: bool


def build_sbar_signals(
    connectome,
    specs: Sequence[SbarSpec],
    *,
    ms_pre: float,
    ms_response: float,
    ms_post: float = 0.0,
    ms_sti=None,
    n_t: Optional[int] = None,
    t_onset: Optional[int] = None,
    delta_ms: float,
    delta_ms_pre: float,
    bar_dist: int,
    multi_bar: bool = True,
    i_baseline: float,
    i_sti: float,
) -> SbarSti:
    """Build sti current for static-bar stis.

    Returns ``i_sti`` with shape ``(B, T, N_nodes)`` where ``B = len(specs)``.
    Peak current ``i_sti`` is for this build (one contrast at a time).
    """
    bar_dist = int(bar_dist)
    multi_bar = bool(multi_bar)
    specs = list(specs)
    i_sti = float(i_sti)
    i_baseline = float(i_baseline)
    sti = sti_hexes(connectome)
    if n_t is None or t_onset is None:
        n_t, t_onset = sbar_n_t(
            ms_pre=ms_pre,
            ms_response=ms_response,
            ms_post=ms_post,
            ms_sti=ms_sti,
            delta_ms=delta_ms,
            delta_ms_pre=delta_ms_pre,
        )
    i_sti_hex = build_i_sti_hex(
        sti, specs, n_t=n_t, bar_dist=bar_dist, multi_bar=multi_bar,
        t_onset=t_onset, delta_ms=delta_ms, ms_sti=ms_sti,
        i_baseline=i_baseline, i_sti=i_sti,
    )
    return SbarSti(
        i_sti=i_sti_nodes_from_hexes(i_sti_hex, sti, connectome.n_node),
        i_sti_hex=i_sti_hex,
        specs=specs,
        n_b=len(specs),
        n_t=n_t,
        t_onset=t_onset,
        i_baseline=i_baseline,
        bar_dist=bar_dist,
        multi_bar=multi_bar,
    )
