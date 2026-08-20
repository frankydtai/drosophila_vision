# -*- coding: utf-8 -*-
"""Bar sti geometry: hex view, bar bounds, fill."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from build_hex import DEG, HEX_AREA, hex_vertices, xy_from_uv, xy_deg_from_uv

# Gap between simultaneous bars, in hex nodes (``bar_radius * DEG``).
BAR_RADIUS = 2


class Bar(Protocol):
    direction: str
    w_deg: float


@dataclass
class Hex:
    """One FAFB sti hex: axial (u,v), hex-step ``(x,y)``, degree ``(x_deg,y_deg)``."""

    u: int
    v: int
    x: float
    y: float
    x_deg: float
    y_deg: float
    vertex_x_degs: np.ndarray
    vertex_y_degs: np.ndarray


def hex_from_uv(u: int, v: int) -> Hex:
    """Build one FAFB sti hex from axial ``(u, v)``."""
    x, y = xy_from_uv(u, v)
    x, y = float(x), float(y)
    x_deg, y_deg = xy_deg_from_uv(u, v)
    x_deg, y_deg = float(x_deg), float(y_deg)
    vertex_degs = hex_vertices(x_deg, y_deg)
    return Hex(
        u=int(u),
        v=int(v),
        x=x,
        y=y,
        x_deg=x_deg,
        y_deg=y_deg,
        vertex_x_degs=np.asarray(vertex_degs[:, 0], dtype=np.float64),
        vertex_y_degs=np.asarray(vertex_degs[:, 1], dtype=np.float64),
    )


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
    """Visible bar ``bounds`` in ``view_deg``; ``None`` when zero visible ``w``."""
    x_deg0, y_deg0, x_deg1, y_deg1 = view_deg
    w_deg = float(bar.w_deg)
    direction = bar.direction
    if direction in ("right", "up"):
        bar_bound0 = max(float(bar_bound), float(bar_bound0))
        bar_bound1 = min(float(bar_bound) + w_deg, float(bar_bound1))
    elif direction in ("left", "down"):
        bar_bound0 = max(float(bar_bound) - w_deg, float(bar_bound0))
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
    bar_radius: int,
    *,
    multi_bar: bool = True,
) -> List[Tuple[float, float]]:
    """``(bar_bound0, bar_bound1)`` along bar ``direction``, covering ``view_deg`` for ``multi_bar``."""
    if int(bar_radius) < 0:
        raise ValueError(f"bar_radius must be >= 0, got {bar_radius!r}")
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
    bar_span_deg = (math.ceil(float(bar.w_deg) / float(DEG)) + int(bar_radius)) * float(DEG)
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


def view_bounds(hexes: Sequence[Hex]) -> Tuple[float, float, float, float]:
    """Sti-view degree bounds from hex vertices."""
    if not hexes:
        return 0.0, 0.0, 0.0, 0.0
    return (
        min(float(sti_hex.vertex_x_degs.min()) for sti_hex in hexes),
        min(float(sti_hex.vertex_y_degs.min()) for sti_hex in hexes),
        max(float(sti_hex.vertex_x_degs.max()) for sti_hex in hexes),
        max(float(sti_hex.vertex_y_degs.max()) for sti_hex in hexes),
    )


@dataclass
class StiHex(Hex):
    """One sti hex on a connectome, with nodes for writing onto ``i_sti``."""

    nodes: np.ndarray


def sti_hexes(connectome) -> List[StiHex]:
    """Sti hexes with sti nodes (one per axial ``(u, v)``)."""
    hex_by_uv: Dict[Tuple[int, int], StiHex] = {}
    for node in connectome.sti_nodes:
        u, v = int(connectome.us[node]), int(connectome.vs[node])
        if (u, v) in hex_by_uv:
            continue
        nodes = connectome.sti_nodes_at_uv(u, v)
        if len(nodes) == 0:
            continue
        sti_hex = hex_from_uv(u, v)
        hex_by_uv[(u, v)] = StiHex(
            u=sti_hex.u, v=sti_hex.v, x=sti_hex.x, y=sti_hex.y,
            x_deg=sti_hex.x_deg, y_deg=sti_hex.y_deg,
            vertex_x_degs=sti_hex.vertex_x_degs, vertex_y_degs=sti_hex.vertex_y_degs,
            nodes=np.asarray(nodes, dtype=np.int64),
        )
    return [hex_by_uv[(u, v)] for u, v in sorted(hex_by_uv)]


def node_us_vs(connectome) -> Tuple[np.ndarray, np.ndarray]:
    """connectome axial ``(u, v)`` per node as int64 numpy."""
    return (
        np.asarray(connectome.us.detach().cpu().numpy(), dtype=np.int64)
        if torch.is_tensor(connectome.us)
        else np.asarray(connectome.us, dtype=np.int64),
        np.asarray(connectome.vs.detach().cpu().numpy(), dtype=np.int64)
        if torch.is_tensor(connectome.vs)
        else np.asarray(connectome.vs, dtype=np.int64),
    )


def sti_hexes_at_xy(hexes, *, at_x=None, at_y=None):
    """Keep network sti hexes at hex-step ``(x, y)`` per ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(hexes)
    return [
        sti_hex for sti_hex in hexes
        if (at_x is None or (
            any(np.isclose(sti_hex.x, float(x), atol=1e-6) for x in at_x)
            if isinstance(at_x, (list, tuple))
            else np.isclose(sti_hex.x, float(at_x), atol=1e-6)
        ))
        and (at_y is None or (
            any(np.isclose(sti_hex.y, float(y), atol=1e-6) for y in at_y)
            if isinstance(at_y, (list, tuple))
            else np.isclose(sti_hex.y, float(at_y), atol=1e-6)
        ))
    ]


def i_sti_nodes_from_hexes(i_sti_hex, hexes, n_node):
    """Map ``(B, T, n_hex)`` i_sti_hex to ``(B, T, n_node)`` by hex->node index."""
    n_b, n_t, _ = i_sti_hex.shape
    i_sti = np.zeros((n_b, n_t, n_node), dtype=np.float64)
    hex_idxs = []
    nodes = []
    for hex_idx, sti_hex in enumerate(hexes):
        for node in np.asarray(sti_hex.nodes).ravel():
            hex_idxs.append(hex_idx)
            nodes.append(int(node))
    if len(hex_idxs):
        hex_idxs = np.asarray(hex_idxs, dtype=np.int64)
        nodes = np.asarray(nodes, dtype=np.int64)
        i_sti[:, :, nodes] = i_sti_hex[:, :, hex_idxs]
    return i_sti
