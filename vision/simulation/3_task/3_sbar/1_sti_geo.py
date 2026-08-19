# -*- coding: utf-8 -*-
"""Static-bar/shared sti geometry: hex view, bar rectangles, clip rect areas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from build_hex import HEX_AREA, hex_vertices, xy_from_uv, xy_deg_from_uv
from network.construction import cost_radius_mask

_HEX_AREA = float(HEX_AREA)


class _BarGeo(Protocol):
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
    hex_xy: np.ndarray


def hex_from_uv(u: int, v: int) -> Hex:
    """Build one FAFB sti hex from axial ``(u, v)``."""
    x, y = xy_from_uv(u, v)
    x, y = float(x), float(y)
    x_deg, y_deg = xy_deg_from_uv(u, v)
    x_deg, y_deg = float(x_deg), float(y_deg)
    return Hex(
        u=int(u),
        v=int(v),
        x=x,
        y=y,
        x_deg=x_deg,
        y_deg=y_deg,
        hex_xy=hex_vertices(x_deg, y_deg),
    )


def _line_intersect(
    px0: float, py0: float, px1: float, py1: float,
    qx0: float, qy0: float, qx1: float, qy1: float,
) -> Tuple[float, float]:
    rx, ry = px1 - px0, py1 - py0
    sx, sy = qx1 - qx0, qy1 - qy0
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return px1, py1
    t = ((qx0 - px0) * sy - (qy0 - py0) * sx) / denom
    return px0 + t * rx, py0 + t * ry


def _clip_halfplane(
    px: np.ndarray,
    py: np.ndarray,
    n: int,
    bound: float,
    *,
    axis: int,
    keep_ge: bool,
    outx: np.ndarray,
    outy: np.ndarray,
) -> int:
    """Sutherland-Hodgman half-plane clip; ``axis`` 0=x / 1=y."""
    m = 0
    prev_x, prev_y = float(px[n - 1]), float(py[n - 1])
    prev_c = prev_x if axis == 0 else prev_y
    prev_in = prev_c >= bound if keep_ge else prev_c <= bound
    if axis == 0:
        qx0, qy0, qx1, qy1 = bound, -1e6, bound, 1e6
    else:
        qx0, qy0, qx1, qy1 = -1e6, bound, 1e6, bound
    for vertex in range(n):
        cur_x, cur_y = float(px[vertex]), float(py[vertex])
        cur_c = cur_x if axis == 0 else cur_y
        cur_in = cur_c >= bound if keep_ge else cur_c <= bound
        if cur_in:
            if not prev_in:
                ix, iy = _line_intersect(prev_x, prev_y, cur_x, cur_y, qx0, qy0, qx1, qy1)
                outx[m], outy[m] = ix, iy
                m += 1
            outx[m], outy[m] = cur_x, cur_y
            m += 1
        elif prev_in:
            ix, iy = _line_intersect(prev_x, prev_y, cur_x, cur_y, qx0, qy0, qx1, qy1)
            outx[m], outy[m] = ix, iy
            m += 1
        prev_x, prev_y, prev_in = cur_x, cur_y, cur_in
    return m


def _poly_area_xy(px: np.ndarray, py: np.ndarray, n: int) -> float:
    if n < 3:
        return 0.0
    x = px[:n]
    y = py[:n]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_rect_area(
    hex_xy: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    hex_area: float = _HEX_AREA,
) -> float:
    """Axis-aligned rect ∩ hex area / hex_area, using fixed buffers (hot path)."""
    w1x = np.empty(12, dtype=np.float64)
    w1y = np.empty(12, dtype=np.float64)
    w1x[:6] = hex_xy[:, 0]
    w1y[:6] = hex_xy[:, 1]
    w2x = np.empty(12, dtype=np.float64)
    w2y = np.empty(12, dtype=np.float64)
    px, py, ox, oy = w1x, w1y, w2x, w2y
    n = 6

    for axis, bound, keep_ge in (
        (0, xmin, True),
        (0, xmax, False),
        (1, ymin, True),
        (1, ymax, False),
    ):
        n = _clip_halfplane(px, py, n, bound, axis=axis, keep_ge=keep_ge, outx=ox, outy=oy)
        if n == 0:
            return 0.0
        px, py, ox, oy = ox, oy, px, py

    return min(1.0, _poly_area_xy(px, py, n) / hex_area)


def clip_rect_areas(
    hex_stack: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> np.ndarray:
    """Clip rect area fraction for every hex against one bar rectangle."""
    n_hex = hex_stack.shape[0]
    hex_clip_rect_areas = np.empty(n_hex, dtype=np.float64)
    for hex_idx in range(n_hex):
        hex_clip_rect_areas[hex_idx] = _clip_rect_area(hex_stack[hex_idx], xmin, ymin, xmax, ymax)
    return hex_clip_rect_areas


def bar_rect_lane_clipped(
    bar: _BarGeo,
    trail: float,
    lane_origin: float,
    lane_pitch: float,
    view_deg: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Bar rectangle clipped to one lane; ``None`` when zero visible w."""
    x0, y0, x1, y1 = view_deg
    w_deg = float(bar.w_deg)
    origin = float(lane_origin)
    lane_end = origin + float(lane_pitch)
    direction = bar.direction
    if direction in ("right", "up"):
        vis_lo = max(float(trail), origin)
        vis_hi = min(float(trail) + w_deg, lane_end)
    elif direction in ("left", "down"):
        vis_lo = max(float(trail) - w_deg, origin)
        vis_hi = min(float(trail), lane_end)
    else:
        raise ValueError(f"unknown direction {direction!r}")
    if vis_lo >= vis_hi - 1e-12:
        return None
    if direction in ("right", "left"):
        return vis_lo, y0, vis_hi, y1
    return x0, vis_lo, x1, vis_hi


def view_bounds(hexes: Sequence[Hex]) -> Tuple[float, float, float, float]:
    """Sti-view degree bounds from hex vertices."""
    if not hexes:
        return 0.0, 0.0, 0.0, 0.0
    xmins = [float(hex.hex_xy[:, 0].min()) for hex in hexes]
    ymins = [float(hex.hex_xy[:, 1].min()) for hex in hexes]
    xmaxs = [float(hex.hex_xy[:, 0].max()) for hex in hexes]
    ymaxs = [float(hex.hex_xy[:, 1].max()) for hex in hexes]
    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


@dataclass
class StiHex(Hex):
    """One sti hex on a connectome, with nodes for writing onto ``i_sti``."""

    nodes: np.ndarray


def sti_hexes(connectome) -> List[StiHex]:
    """Sti hexes with sti nodes (one per axial ``(u, v)``)."""
    by_uv: Dict[Tuple[int, int], StiHex] = {}
    for node in connectome.sti_nodes:
        u, v = int(connectome.us[node]), int(connectome.vs[node])
        key = (u, v)
        if key in by_uv:
            continue
        nodes = connectome.sti_nodes_at_uv(u, v)
        if len(nodes) == 0:
            continue
        base = hex_from_uv(key[0], key[1])
        by_uv[key] = StiHex(
            u=base.u, v=base.v, x=base.x, y=base.y,
            x_deg=base.x_deg, y_deg=base.y_deg, hex_xy=base.hex_xy,
            nodes=np.asarray(nodes, dtype=np.int64),
        )
    return [by_uv[k] for k in sorted(by_uv)]


def mbar_cost_hexes(connectome, cost_radius=None) -> List[StiHex]:
    """Sti hexes used for moving-bar cost (optional central hex disc)."""
    hexes = sti_hexes(connectome)
    if cost_radius is None:
        return hexes
    return [hex for hex in hexes if cost_radius_mask(hex.u, hex.v, cost_radius)]


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


def filter_sti_hexes(hexes, *, at_x=None, at_y=None, tol=1e-6):
    """Keep network sti hexes at hex-step ``(x, y)`` per ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(hexes)
    return [
        hex for hex in hexes
        if (at_x is None or (
            any(np.isclose(hex.x, float(x), atol=tol) for x in at_x)
            if isinstance(at_x, (list, tuple))
            else np.isclose(hex.x, float(at_x), atol=tol)
        ))
        and (at_y is None or (
            any(np.isclose(hex.y, float(y), atol=tol) for y in at_y)
            if isinstance(at_y, (list, tuple))
            else np.isclose(hex.y, float(at_y), atol=tol)
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
