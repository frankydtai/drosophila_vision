# -*- coding: utf-8 -*-
"""Lattice <-> node lookups for connectome multi-column training.

Bridges the pure hex geometry in ``column_mapper`` (tile centres, ring/tile offsets,
shifts) to the concrete nodes of a loaded :class:`connectome_network.Connectome`:

  - :func:`col2photo` -- the stimulus (photoreceptor) units on a column.
  - :func:`col2fit`   -- the fit-cell units of a given type on a column.
  - :func:`build_tiling` -- a :class:`Tiling`: tile centres x member columns,
    reusing ``column_mapper.tile_centers`` / ``tile_offsets``.
  - :func:`shifted_photoreceptors` -- stimulus units for each of the 7 sub-tile
    shifts (the tile centre + its 6 neighbours).

The fit cell vocabulary is the same 13 types the 5-column model fits
(``Medulla_Library.cell_list``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import connectome_path  # noqa: F401  (adds the FAFB connectome folder to sys.path)

import column_mapper  # noqa: E402

from Medulla_Library import cell_list as _CELL_LIST  # noqa: E402

FIT_CELL_TYPES: List[str] = [str(c) for c in _CELL_LIST]  # 13 fit types


def euclid_hex_dist(du: int, dv: int) -> float:
    """Euclidean distance (in column units) between two axial cells.

    Nearest neighbours are at distance 1; the extent-2 ring splits into corners
    at r=2 ((2,0),(2,-2),...) and edge midpoints at r=sqrt(3) ((2,-1),(1,1),...).
    """
    return math.sqrt(du * du + du * dv + dv * dv)


def unit_type_names(C) -> np.ndarray:
    """(n_units,) array of each unit's cell-type NAME."""
    return np.asarray(C.type_names)[C.node_type.detach().cpu().numpy()]


def col2photo(C, u: int, v: int) -> np.ndarray:
    """Stimulus (photoreceptor / input) unit indices on column (u, v)."""
    return C.input_units_at(int(u), int(v))


def col2fit(C, u: int, v: int, fit_type: str, names: np.ndarray = None) -> np.ndarray:
    """Unit indices of cell type ``fit_type`` on column (u, v)."""
    if names is None:
        names = unit_type_names(C)
    return np.where((C.u == int(u)) & (C.v == int(v)) & (names == fit_type))[0]


@dataclass
class Tiling:
    """Tile centres x member columns over a loaded connectome.

    centers:  list of (u, v) tile-centre axial coords.
    members:  list of (du, dv) member offsets shared by every tile (tile_offsets).
    shifts:   list of (du, dv) sub-tile shifts (7: centre + 6 neighbours).
    """

    centers: List[Tuple[int, int]]
    members: List[Tuple[int, int]]
    shifts: List[Tuple[int, int]]
    tile_extent: int
    share_edges: bool

    def member_columns(self, center: Tuple[int, int]) -> List[Tuple[int, int]]:
        cu, cv = center
        return [(cu + du, cv + dv) for du, dv in self.members]


def _graph_extent(C, tile_extent: int) -> int:
    """Hex-disc radius of connectome ``C``.

    ``meta["extent"] >= 0`` is a real crop radius and used as-is. ``< 0`` (or
    missing) means no crop, so the radius is the largest ``hex_radius`` over the
    positioned columns (column_id >= 0); falls back to ``tile_extent`` if none.
    """
    meta_extent = int(C.meta.get("extent", -1))
    if meta_extent >= 0:
        return meta_extent
    positioned = C.column_id >= 0
    radii = [
        column_mapper.hex_radius(int(u), int(v))
        for u, v in zip(C.u[positioned], C.v[positioned])
    ]
    return max(radii) if radii else tile_extent


def build_tiling(
    C,
    tile_extent: int = 2,
    share_edges: bool = False,
    single_tile: bool = None,
) -> Tiling:
    """Build a :class:`Tiling` for connectome ``C``.

    If ``single_tile`` (default: auto when the graph's own extent <= tile_extent),
    the whole graph is one tile centred at (0, 0) -- the right case for an
    already-cropped extent-2 sub-graph. Otherwise tiles come from
    ``column_mapper.tile_centers`` over the graph's extent (31 disjoint / 43 sharing).

    The graph extent is ``meta["extent"]`` when it is a real crop radius (>= 0);
    a value < 0 means "no crop", so the extent is derived from the actual radius
    spanned by the positioned columns (otherwise the full graph would collapse to
    a single tile).
    """
    graph_extent = _graph_extent(C, tile_extent)
    if single_tile is None:
        single_tile = graph_extent <= tile_extent
    members = [(int(du), int(dv)) for du, dv in column_mapper.tile_offsets(tile_extent)]
    shifts = [(int(du), int(dv)) for du, dv in column_mapper.shift_offsets()]
    if single_tile:
        centers = [(0, 0)]
    else:
        centers = [
            (int(cu), int(cv))
            for cu, cv in column_mapper.tile_centers(
                extent=graph_extent,
                tile_extent=tile_extent,
                share_edges=share_edges,
            )
        ]
    return Tiling(centers, members, shifts, tile_extent, share_edges)


def shifted_photoreceptors(C, center: Tuple[int, int], shifts) -> List[np.ndarray]:
    """For a tile centre, the stimulus units at centre+shift for each shift."""
    cu, cv = center
    return [col2photo(C, cu + du, cv + dv) for du, dv in shifts]
