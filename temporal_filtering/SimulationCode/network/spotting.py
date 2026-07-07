# -*- coding: utf-8 -*-
"""Lattice <-> node lookups for connectome multi-column training.

Bridges the pure hex geometry in ``column_mapper`` (spot centres, ring/spot offsets,
shifts) to the concrete nodes of a loaded :class:`network.construction.Network`:

  - :func:`col2sti` -- the stimulus (photoreceptor) units on a column.
  - :func:`col2fit`   -- the fit-cell units of a given type on a column.
  - :func:`build_spotting` -- a :class:`Spotting`: spot centres x member columns,
    reusing ``column_mapper.spot_centers`` / ``spot_offsets``.
  - :func:`shifted_photoreceptors` -- stimulus units for each configured sub-spot
    shift around each spot centre.

The fit cell vocabulary is the same 13 types the 5-column model fits
(``Medulla_Library.cell_list``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import network_bootstrap  # noqa: F401

import column_mapper

from Medulla_Library import cell_list as _CELL_LIST

FIT_CELL_TYPES: List[str] = [str(c) for c in _CELL_LIST]  # 13 fit types
CENTER_COLUMN_UV = (0, 0)


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in column units) between two axial cells.

    Nearest neighbours are at distance 1; the extent-2 ring splits into corners
    at r=2 ((2,0),(2,-2),...) and edge midpoints at r=sqrt(3) ((2,-1),(1,1),...).
    """
    return math.sqrt(du * du + du * dv + dv * dv)


def unit_type_names(C) -> np.ndarray:
    """(n_units,) array of each unit's cell-type NAME."""
    return np.asarray(C.type_names)[C.node_type.detach().cpu().numpy()]


def col2sti(C, u: int, v: int) -> np.ndarray:
    """Stimulus (photoreceptor / input) unit indices on column (u, v)."""
    return C.input_units_at(int(u), int(v))


def col2fit(C, u: int, v: int, fit_type: str, names: np.ndarray = None) -> np.ndarray:
    """Unit indices of cell type ``fit_type`` on column (u, v)."""
    if names is None:
        names = unit_type_names(C)
    return np.where((C.u == int(u)) & (C.v == int(v)) & (names == fit_type))[0]


@dataclass
class Spotting:
    """Tile centres x member columns over a loaded connectome.

    centers:  list of (u, v) spot-centre axial coords.
    members:  list of (du, dv) member offsets shared by every spot (spot_offsets).
    shifts:   list of (du, dv) sub-spot shifts from ``spot_offsets(shift_extent)``.
    """

    centers: List[Tuple[int, int]]
    members: List[Tuple[int, int]]
    shifts: List[Tuple[int, int]]
    spot_extent: int
    share_edges: bool

    def member_columns(self, center: Tuple[int, int]) -> List[Tuple[int, int]]:
        cu, cv = center
        return [(cu + du, cv + dv) for du, dv in self.members]


def spot_stimulus_batches(spotting: Spotting) -> List[Tuple[int, int, Tuple[int, int]]]:
    """One batch per (spot centre, shift): ``(stim_u, stim_v, center)``."""
    batches = []
    for center in spotting.centers:
        for du, dv in spotting.shifts:
            batches.append((center[0] + du, center[1] + dv, center))
    return batches


def spotting_from_opts(
    C,
    spot_extent: int = 2,
    share_edges: bool = False,
    shift_extent: int = 0,
    single_spot: Optional[bool] = None,
) -> Spotting:
    """Build :class:`Spotting` with configurable sub-spot shift radius."""
    spotting = build_spotting(C, spot_extent, share_edges, single_spot)
    spotting.shifts = [(int(du), int(dv)) for du, dv in column_mapper.spot_offsets(int(shift_extent))]
    return spotting


def spotting_from_stimulus_opts(C, opts: Dict) -> Spotting:
    """``train_opts`` spot stimulus dict → :class:`Spotting`."""
    return spotting_from_opts(
        C,
        spot_extent=int(opts.get("spot_extent", 2)),
        share_edges=bool(opts.get("share_edges", False)),
        shift_extent=int(opts.get("shift_extent", 0)),
    )


def _uv_arrays(C):
    u = C.u.detach().cpu().numpy() if hasattr(C.u, "detach") else np.asarray(C.u)
    v = C.v.detach().cpu().numpy() if hasattr(C.v, "detach") else np.asarray(C.v)
    return u, v


def unit_ring_layout(C, batches, units=None):
    """Per (batch, unit): batch_idx, unit_idx, stim-centred radius, type_idx."""
    u_all, v_all = _uv_arrays(C)
    if units is None:
        units = np.arange(C.n_units, dtype=np.int64)
    else:
        units = np.asarray(units, dtype=np.int64)
    type_all = (
        C.node_type.detach().cpu().numpy()
        if hasattr(C.node_type, "detach") else np.asarray(C.node_type)
    )
    batch_idx, unit_idx, radius, type_idx = [], [], [], []
    for b, (su, sv, _center) in enumerate(batches):
        for u in units:
            batch_idx.append(b)
            unit_idx.append(int(u))
            radius.append(euclid_hex_dist(int(u_all[u]) - su, int(v_all[u]) - sv))
            type_idx.append(int(type_all[u]))
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
    )


def _graph_extent(C, spot_extent: int) -> int:
    """Hex-disc radius of connectome ``C``.

    ``meta["extent"] >= 0`` is a real crop radius and used as-is. ``< 0`` (or
    missing) means no crop, so the radius is the largest ``hex_radius`` over the
    positioned columns (column_id >= 0); falls back to ``spot_extent`` if none.
    """
    meta_extent = int(C.meta.get("extent", -1))
    if meta_extent >= 0:
        return meta_extent
    positioned = C.column_id >= 0
    radii = [
        column_mapper.hex_radius(int(u), int(v))
        for u, v in zip(C.u[positioned], C.v[positioned])
    ]
    return max(radii) if radii else spot_extent


def build_spotting(
    C,
    spot_extent: int = 2,
    share_edges: bool = False,
    single_spot: bool = None,
) -> Spotting:
    """Build a :class:`Spotting` for connectome ``C``.

    If ``single_spot`` (default: auto when the graph's own extent <= spot_extent),
    the whole graph is one spot centred at (0, 0) -- the right case for an
    already-cropped extent-2 sub-graph. Otherwise spots come from
    ``column_mapper.spot_centers`` over the graph's extent (31 disjoint / 43 sharing).

    The graph extent is ``meta["extent"]`` when it is a real crop radius (>= 0);
    a value < 0 means "no crop", so the extent is derived from the actual radius
    spanned by the positioned columns (otherwise the full graph would collapse to
    a single spot).
    """
    graph_extent = _graph_extent(C, spot_extent)
    if single_spot is None:
        single_spot = graph_extent <= spot_extent
    members = [(int(du), int(dv)) for du, dv in column_mapper.spot_offsets(spot_extent)]
    shifts = [(int(du), int(dv)) for du, dv in column_mapper.shift_offsets()]
    if single_spot:
        centers = [(0, 0)]
    else:
        centers = [
            (int(cu), int(cv))
            for cu, cv in column_mapper.spot_centers(
                extent=graph_extent,
                spot_extent=spot_extent,
                share_edges=share_edges,
            )
        ]
    return Spotting(centers, members, shifts, spot_extent, share_edges)


def shifted_photoreceptors(C, center: Tuple[int, int], shifts) -> List[np.ndarray]:
    """For a spot centre, the stimulus units at centre+shift for each shift."""
    cu, cv = center
    return [col2sti(C, cu + du, cv + dv) for du, dv in shifts]
