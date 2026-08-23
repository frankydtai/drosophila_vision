# -*- coding: utf-8 -*-
"""Moving-bar geometry: bar bounds and hex fill overlap."""
from __future__ import annotations

import math
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from build_hex import DEG, HEX_AREA, HEX_PATCH_RADIUS


class Bar(Protocol):
    direction: str
    bar_w_deg: float


def _vertex_from_fill(
    x_pre: float, y_pre: float, x_post: float, y_post: float, clip_deg: float, *,
    clip_x: bool,
) -> Tuple[float, float]:
    if clip_x:
        if abs(x_post - x_pre) < 1e-12:
            return x_post, y_post
        return clip_deg, y_pre + (clip_deg - x_pre) / (x_post - x_pre) * (y_post - y_pre)
    if abs(y_post - y_pre) < 1e-12:
        return x_post, y_post
    return x_pre + (clip_deg - y_pre) / (y_post - y_pre) * (x_post - x_pre), clip_deg


def _fill_vertex_degs(
    vertex_x_degs_pre: np.ndarray,
    vertex_y_degs_pre: np.ndarray,
    n_vertex: int,
    clip_deg: float,
    vertex_x_degs_post: np.ndarray,
    vertex_y_degs_post: np.ndarray,
    *,
    clip_x: bool,
    clip_ge: bool,
) -> int:
    n_vertex_post = 0
    x_pre, y_pre = (
        float(vertex_x_degs_pre[n_vertex - 1]), float(vertex_y_degs_pre[n_vertex - 1])
    )
    if clip_x:
        active_pre = x_pre >= clip_deg if clip_ge else x_pre <= clip_deg
    else:
        active_pre = y_pre >= clip_deg if clip_ge else y_pre <= clip_deg
    for vertex in range(n_vertex):
        x_post, y_post = (
            float(vertex_x_degs_pre[vertex]), float(vertex_y_degs_pre[vertex])
        )
        if clip_x:
            active_post = x_post >= clip_deg if clip_ge else x_post <= clip_deg
        else:
            active_post = y_post >= clip_deg if clip_ge else y_post <= clip_deg
        if active_post:
            if not active_pre:
                vertex_x_degs_post[n_vertex_post], vertex_y_degs_post[n_vertex_post] = (
                    _vertex_from_fill(
                        x_pre, y_pre, x_post, y_post, clip_deg, clip_x=clip_x,
                    )
                )
                n_vertex_post += 1
            vertex_x_degs_post[n_vertex_post], vertex_y_degs_post[n_vertex_post] = x_post, y_post
            n_vertex_post += 1
        elif active_pre:
            vertex_x_degs_post[n_vertex_post], vertex_y_degs_post[n_vertex_post] = (
                _vertex_from_fill(
                    x_pre, y_pre, x_post, y_post, clip_deg, clip_x=clip_x,
                )
            )
            n_vertex_post += 1
        x_pre, y_pre, active_pre = x_post, y_post, active_post
    return n_vertex_post


def _fill(
    vertex_x_degs: np.ndarray,
    vertex_y_degs: np.ndarray,
    x_deg0: float,
    y_deg0: float,
    x_deg1: float,
    y_deg1: float,
) -> float:
    """Bar bounds ∩ hex, divided by ``HEX_AREA``, in [0, 1]."""
    vertex_x_degs_pre = np.empty(12, dtype=np.float64)
    vertex_y_degs_pre = np.empty(12, dtype=np.float64)
    vertex_x_degs_post = np.empty(12, dtype=np.float64)
    vertex_y_degs_post = np.empty(12, dtype=np.float64)
    vertex_x_degs_pre[:6] = vertex_x_degs
    vertex_y_degs_pre[:6] = vertex_y_degs
    n_vertex = 6
    for clip_x, clip_ge, clip_deg in (
        (True, True, x_deg0),
        (True, False, x_deg1),
        (False, True, y_deg0),
        (False, False, y_deg1),
    ):
        n_vertex = _fill_vertex_degs(
            vertex_x_degs_pre, vertex_y_degs_pre, n_vertex, clip_deg,
            vertex_x_degs_post, vertex_y_degs_post,
            clip_x=clip_x, clip_ge=clip_ge,
        )
        if n_vertex == 0:
            return 0.0
        vertex_x_degs_pre, vertex_y_degs_pre, vertex_x_degs_post, vertex_y_degs_post = (
            vertex_x_degs_post, vertex_y_degs_post, vertex_x_degs_pre, vertex_y_degs_pre,
        )
    if n_vertex < 3:
        return 0.0
    return min(
        1.0,
        0.5 * abs(float(
            np.dot(vertex_x_degs_pre[:n_vertex], np.roll(vertex_y_degs_pre[:n_vertex], -1))
            - np.dot(vertex_y_degs_pre[:n_vertex], np.roll(vertex_x_degs_pre[:n_vertex], -1))
        )) / float(HEX_AREA),
    )


def bar_bounds(
    bar: Bar,
    bar_bound: float,
    view_deg: Tuple[float, float, float, float],
    bar_bound0: float,
    bar_bound1: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Visible bar ``bounds`` in ``view_deg``; ``None`` when zero visible ``bar_w``."""
    x_deg0, y_deg0, x_deg1, y_deg1 = view_deg
    bar_w_deg = float(bar.bar_w_deg)
    direction = bar.direction
    if direction in ("right", "up"):
        bar_bound0 = max(float(bar_bound), float(bar_bound0))
        bar_bound1 = min(float(bar_bound) + bar_w_deg, float(bar_bound1))
    elif direction in ("left", "down"):
        bar_bound0 = max(float(bar_bound) - bar_w_deg, float(bar_bound0))
        bar_bound1 = min(float(bar_bound), float(bar_bound1))
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if bar_bound0 >= bar_bound1 - 1e-12:
        return None
    if direction in ("right", "left"):
        return bar_bound0, y_deg0, bar_bound1, y_deg1
    return x_deg0, bar_bound0, x_deg1, bar_bound1


def bar_bound0_bar_bound1s(
    bar: Bar,
    view_deg: Tuple[float, float, float, float],
    bar_dist: int,
    *,
    multi_bar: bool = True,
) -> List[Tuple[float, float]]:
    """``(bar_bound0, bar_bound1)`` along bar ``direction``, covering ``view_deg`` for ``multi_bar``."""
    if int(bar_dist) < 0:
        raise ValueError(f"bar_dist must be >= 0, got {bar_dist!r}")
    x_deg0, y_deg0, x_deg1, y_deg1 = view_deg
    if bar.direction in ("right", "left"):
        if not bool(multi_bar):
            return [(float(x_deg0), float(x_deg1))]
        if bar.direction == "right":
            view_bar_bound0, view_bar_bound1 = float(x_deg0), float(x_deg1)
        else:
            view_bar_bound0, view_bar_bound1 = float(x_deg1), float(x_deg0)
    elif bar.direction in ("up", "down"):
        if not bool(multi_bar):
            return [(float(y_deg0), float(y_deg1))]
        if bar.direction == "up":
            view_bar_bound0, view_bar_bound1 = float(y_deg0), float(y_deg1)
        else:
            view_bar_bound0, view_bar_bound1 = float(y_deg1), float(y_deg0)
    else:
        raise ValueError(f"unknown direction {bar.direction!r}")
    bar_span_deg = (math.ceil(float(bar.bar_w_deg) / float(DEG)) + int(bar_dist)) * float(DEG)
    bar_bound0_bar_bound1s: List[Tuple[float, float]] = []
    if float(view_bar_bound0) < float(view_bar_bound1):
        bar_bound0 = float(view_bar_bound0)
        while bar_bound0 < float(view_bar_bound1) - 1e-9:
            bar_bound0_bar_bound1s.append((
                bar_bound0, min(bar_bound0 + bar_span_deg, float(view_bar_bound1)),
            ))
            bar_bound0 += bar_span_deg
    else:
        bar_bound0 = float(view_bar_bound0) - bar_span_deg
        while bar_bound0 + bar_span_deg > float(view_bar_bound1) + 1e-9:
            if bar_bound0 < float(view_bar_bound1) - 1e-9:
                bar_bound0_bar_bound1s.append((
                    float(view_bar_bound1), bar_bound0 + bar_span_deg,
                ))
                break
            bar_bound0_bar_bound1s.append((bar_bound0, bar_bound0 + bar_span_deg))
            bar_bound0 -= bar_span_deg
    if not bar_bound0_bar_bound1s:
        bar_bound0 = min(float(view_bar_bound0), float(view_bar_bound1))
        bar_bound0_bar_bound1s.append((
            bar_bound0,
            bar_bound0 + min(bar_span_deg, abs(float(view_bar_bound1) - float(view_bar_bound0))),
        ))
    return bar_bound0_bar_bound1s


def view_bounds(hexes: Sequence) -> Tuple[float, float, float, float]:
    """Sti-view degree bounds from hex centres ± ``HEX_PATCH_RADIUS``."""
    if not hexes:
        return 0.0, 0.0, 0.0, 0.0
    pad = float(HEX_PATCH_RADIUS)
    return (
        min(float(hex.x) * float(DEG) for hex in hexes) - pad,
        min(float(hex.y) * float(DEG) for hex in hexes) - pad,
        max(float(hex.x) * float(DEG) for hex in hexes) + pad,
        max(float(hex.y) * float(DEG) for hex in hexes) + pad,
    )


def sbar_line_hex_mask(
    hexes: Sequence,
    direction: str,
    bar_dist: int,
    *,
    multi_bar: bool = True,
) -> np.ndarray:
    """Static-bar width-1 lines: binary ``hex_mask`` ``(n_hex,)`` at ``bar_span``."""
    n_hex = len(hexes)
    hex_mask = np.zeros(n_hex, dtype=np.float64)
    if n_hex == 0:
        return hex_mask
    bar_span = 1 + int(bar_dist)
    if direction in ("right", "left"):
        lit = sorted({float(hex.x) for hex in hexes})
        if direction == "left":
            lit = lit[::-1]
        if not multi_bar:
            lit = lit[:1]
        else:
            lit = lit[0::bar_span]
        for hex_idx, hex in enumerate(hexes):
            if float(hex.x) in lit:
                hex_mask[hex_idx] = 1.0
    elif direction in ("up", "down"):
        lit = sorted({float(hex.y) for hex in hexes})
        if direction == "down":
            lit = lit[::-1]
        if not multi_bar:
            lit = lit[:1]
        else:
            lit = lit[0::bar_span]
        for hex_idx, hex in enumerate(hexes):
            if float(hex.y) in lit:
                hex_mask[hex_idx] = 1.0
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return hex_mask
