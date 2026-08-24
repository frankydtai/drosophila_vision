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
    sti_hexes,
)
from task.spread.sti_spec import sti_mask

__all__ = (
    "SbarSpec",
    "SbarSti",
    "build_i_sti_hex",
    "build_sbar_signals",
    "gruntman_sbar_specs",
    "sbar_n_t",
)


@dataclass(frozen=True)
class SbarSpec:
    direction: str
    contrast: str

    @property
    def token(self) -> str:
        return f"{self.direction}_{self.contrast}_w1"


def gruntman_sbar_specs(
    *,
    contrasts: Sequence[str],
    bar_directions: Sequence[str],
) -> List[SbarSpec]:
    """Whole-view static-bar conditions (width-1 hex lines) for ``contrasts``."""
    return [
        SbarSpec(direction=direction, contrast=contrast)
        for direction in bar_directions
        for contrast in contrasts
    ]


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

    by_direction: dict[str, List[int]] = {}
    for b, spec in enumerate(specs):
        by_direction.setdefault(spec.direction, []).append(b)

    for direction, bs in by_direction.items():
        hex_mask = sbar_line_hex_mask(
            hexes, direction, bar_dist, multi_bar=multi_bar,
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
