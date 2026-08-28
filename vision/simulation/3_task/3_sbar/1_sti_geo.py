# -*- coding: utf-8 -*-
"""Hex sti geometry: connectome sti hex mapping (no bar math)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from build_hex import radius_hexes, xy_from_uv


@dataclass
class Hex:
    """One FAFB sti hex: axial ``(u, v)`` and hex-step ``(x, y)``."""

    u: int
    v: int
    x: float
    y: float


def hex_from_uv(u: int, v: int) -> Hex:
    """Build one FAFB sti hex from axial ``(u, v)``."""
    x, y = xy_from_uv(u, v)
    return Hex(u=int(u), v=int(v), x=float(x), y=float(y))


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


def i_sti_nodes_from_hexes(i_sti_hex, hexes, n_node, *, fill=0.0):
    """Map ``(B, T, n_hex)`` i_sti_hex to ``(B, T, n_node)`` by hex->node index."""
    i_sti = np.full(
        (i_sti_hex.shape[0], i_sti_hex.shape[1], n_node),
        float(fill), dtype=np.float64,
    )
    hex_idxs = []
    nodes = []
    for hex_idx, sti_hex in enumerate(hexes):
        for node in np.asarray(sti_hex.nodes).ravel():
            hex_idxs.append(hex_idx)
            nodes.append(int(node))
    if len(hex_idxs):
        i_sti[:, :, np.asarray(nodes, dtype=np.int64)] = i_sti_hex[:, :, np.asarray(hex_idxs, dtype=np.int64)]
    return i_sti


def sbar_line_hex_mask(
    hexes: Sequence,
    direction: str,
    bar_dist: int,
    *,
    multi_bar: bool = True,
    shift_mid: float = 0.0,
) -> np.ndarray:
    """Static-bar width-1 lines: binary ``hex_mask`` ``(n_hex,)`` on ``bar_dist`` steps."""
    n_hex = len(hexes)
    hex_mask = np.zeros(n_hex, dtype=np.float64)
    if n_hex == 0:
        return hex_mask
    lit = sbar_line_mids(
        hexes, direction, bar_dist, multi_bar=multi_bar,
        shift_mid=shift_mid,
    )
    if direction in ("right", "left"):
        for hex_idx, hex in enumerate(hexes):
            if float(hex.x) in lit:
                hex_mask[hex_idx] = 1.0
    elif direction in ("up", "down"):
        for hex_idx, hex in enumerate(hexes):
            if float(hex.y) in lit:
                hex_mask[hex_idx] = 1.0
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return hex_mask


def sbar_line_mids(
    hexes: Sequence,
    direction: str,
    bar_dist: int,
    *,
    multi_bar: bool = True,
    shift_mid: float = 0.0,
) -> List[float]:
    """Axis coordinates of static bars; single-bar uses the central line."""
    if direction in ("right", "left"):
        lit = sorted({float(hex.x) for hex in hexes})
        reverse = direction == "left"
    elif direction in ("up", "down"):
        lit = sorted({float(hex.y) for hex in hexes})
        reverse = direction == "down"
    else:
        raise ValueError(f"unknown direction {direction!r}")
    available = tuple(lit)
    if not multi_bar:
        base = [min(lit, key=lambda value: (abs(value), value))] if lit else []
        shifted = [float(value + shift_mid) for value in base]
        return [
            value for value in shifted
            if any(np.isclose(value, candidate, atol=1e-9) for candidate in available)
        ]
    if reverse:
        lit = lit[::-1]
    bar_dist = int(bar_dist)
    if bar_dist > 0:
        lit = [
            value for value in lit
            if abs(value - bar_dist * round(value / bar_dist)) < 1e-9
        ]
    shifted = [float(value + shift_mid) for value in lit]
    return [
        value for value in shifted
        if any(np.isclose(value, candidate, atol=1e-9) for candidate in available)
    ]


def sbar_shift_mids(direction: str, shift_radius: int) -> List[float]:
    """Unique bar-normal projections of axial shifts within ``shift_radius``."""
    shift_radius = int(shift_radius)
    if shift_radius < 0:
        raise ValueError(f"sbar shift_radius must be >= 0, got {shift_radius}")
    shifts = radius_hexes(shift_radius)
    if direction in ("right", "left"):
        values = {float(dv) for du, dv in shifts}
    elif direction in ("up", "down"):
        values = {float(du) + float(dv) / 2.0 for du, dv in shifts}
    else:
        raise ValueError(f"unknown direction {direction!r}")
    return sorted(values)
