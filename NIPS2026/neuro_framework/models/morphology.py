"""
SWC Morphology Loader
=====================
Reads SWC files and builds compartmental electrical models compatible
with Jaxley conventions (truncated-cone areas, resistive-load axial
coupling, same unit system).

This module is used by the multi-compartment dynamics models.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class MorphologyGraph:
    """Compartmental electrical model built from an SWC file.

    Attributes
    ----------
    n_comp : int
        Total number of compartments across all branches.
    areas, volumes : ndarray  (n_comp,)
        Surface area (μm²) and volume (μm³) per compartment.
    resistive_load_in, resistive_load_out : ndarray  (n_comp,)
        Resistive load from each compartment half (μm⁻¹).
    radii, lengths : ndarray  (n_comp,)
        Average radius (μm) and path length (μm) per compartment.
    comp_edges : list of (int, int)
        Directed edges between compartments (bidirectional pairs).
    branch_parents : list of int
        Parent branch index for each branch (-1 for root).
    n_branches : int
        Number of morphological branches.
    """

    def __init__(
        self,
        n_comp: int,
        areas: np.ndarray,
        volumes: np.ndarray,
        resistive_load_in: np.ndarray,
        resistive_load_out: np.ndarray,
        radii: np.ndarray,
        lengths: np.ndarray,
        comp_edges: List[Tuple[int, int]],
        branch_parents: List[int],
        n_branches: int,
    ):
        self.n_comp = n_comp
        self.areas = areas
        self.volumes = volumes
        self.resistive_load_in = resistive_load_in
        self.resistive_load_out = resistive_load_out
        self.radii = radii
        self.lengths = lengths
        self.comp_edges = comp_edges
        self.branch_parents = branch_parents
        self.n_branches = n_branches

    @classmethod
    def from_swc(
        cls,
        swc_path: str,
        ncomp: int = 1,
        min_radius: float = 0.1,
        scale: float = 1.0,
    ) -> "MorphologyGraph":
        """Load an SWC file and compartmentalise each branch.

        Parameters
        ----------
        scale:
            Multiplies x/y/z and radius values before building the morphology.
            Use this when an SWC is stored in voxel units and you want microns.
        """
        nodes = _load_swc(swc_path)
        if scale != 1.0:
            for node in nodes.values():
                node["x"] *= scale
                node["y"] *= scale
                node["z"] *= scale
                node["r"] *= scale
        return cls.from_nodes(nodes, ncomp=ncomp, min_radius=min_radius)

    @classmethod
    def from_swc_rows(
        cls,
        swc_rows,
        ncomp: int = 1,
        min_radius: float = 0.1,
        scale: float = 1.0,
    ) -> "MorphologyGraph":
        """Build directly from in-memory SWC rows.

        Parameters
        ----------
        swc_rows : iterable
            Rows with fields/columns: id, type, x, y, z, r, parent.
        scale:
            Multiplies x/y/z and radius values before building the morphology.
        """
        nodes = _rows_to_nodes(swc_rows)
        if scale != 1.0:
            for node in nodes.values():
                node["x"] *= scale
                node["y"] *= scale
                node["z"] *= scale
                node["r"] *= scale
        return cls.from_nodes(nodes, ncomp=ncomp, min_radius=min_radius)

    @classmethod
    def from_nodes(
        cls,
        nodes: Dict[int, Dict],
        ncomp: int = 1,
        min_radius: float = 0.1,
    ) -> "MorphologyGraph":
        """Build a morphology graph from parsed SWC nodes."""
        root = _find_root(nodes)
        branches, branch_parent_idx = _trace_branches(nodes, root)

        all_areas: List[float] = []
        all_volumes: List[float] = []
        all_r_in: List[float] = []
        all_r_out: List[float] = []
        all_radii: List[float] = []
        all_lengths: List[float] = []
        comp_edges: List[Tuple[int, int]] = []
        comp_offset = 0

        for b_idx, branch_node_ids in enumerate(branches):
            xyzr = np.array(
                [
                    [
                        nodes[nid]["x"],
                        nodes[nid]["y"],
                        nodes[nid]["z"],
                        max(nodes[nid]["r"], min_radius),
                    ]
                    for nid in branch_node_ids
                ]
            )

            xyzr_per_comp = _split_xyzr(xyzr, ncomp)
            for comp_xyzr in xyzr_per_comp:
                r, a, v, rl_in, rl_out = _comp_morph_attrs(comp_xyzr, min_radius, ncomp)
                all_radii.append(r)
                all_areas.append(a)
                all_volumes.append(v)
                all_r_in.append(rl_in)
                all_r_out.append(rl_out)

                positions = comp_xyzr[:, :3]
                seg_len = (
                    np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                    if len(positions) > 1
                    else 2 * comp_xyzr[0, 3] / ncomp
                )
                all_lengths.append(seg_len)

            for c in range(ncomp - 1):
                ci, cj = comp_offset + c, comp_offset + c + 1
                comp_edges.append((ci, cj))
                comp_edges.append((cj, ci))

            parent_b = branch_parent_idx[b_idx]
            if parent_b >= 0:
                parent_last = parent_b * ncomp + (ncomp - 1)
                child_first = comp_offset
                comp_edges.append((parent_last, child_first))
                comp_edges.append((child_first, parent_last))

            comp_offset += ncomp

        return cls(
            n_comp=comp_offset,
            areas=np.array(all_areas),
            volumes=np.array(all_volumes),
            resistive_load_in=np.array(all_r_in),
            resistive_load_out=np.array(all_r_out),
            radii=np.array(all_radii),
            lengths=np.array(all_lengths),
            comp_edges=comp_edges,
            branch_parents=branch_parent_idx,
            n_branches=len(branches),
        )


# -----------------------------------------------------------------------
# SWC helpers
# -----------------------------------------------------------------------


def _load_swc(path: str) -> Dict[int, Dict]:
    nodes: Dict[int, Dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            nodes[nid] = {
                "id": nid,
                "type": int(parts[1]),
                "x": float(parts[2]),
                "y": float(parts[3]),
                "z": float(parts[4]),
                "r": float(parts[5]),
                "parent": int(parts[6]),
                "children": [],
            }
    for nid, n in nodes.items():
        pid = n["parent"]
        if pid != -1 and pid in nodes:
            nodes[pid]["children"].append(nid)
    return nodes


def _rows_to_nodes(swc_rows) -> Dict[int, Dict]:
    nodes: Dict[int, Dict] = {}
    for row in swc_rows:
        if isinstance(row, np.void) and getattr(row, "dtype", None) is not None:
            nid = int(row["id"])
            nodes[nid] = {
                "id": nid,
                "type": int(row["type"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "r": float(row["r"]),
                "parent": int(row["parent"]),
                "children": [],
            }
        else:
            nid = int(getattr(row, "id"))
            nodes[nid] = {
                "id": nid,
                "type": int(getattr(row, "type")),
                "x": float(getattr(row, "x")),
                "y": float(getattr(row, "y")),
                "z": float(getattr(row, "z")),
                "r": float(getattr(row, "r")),
                "parent": int(getattr(row, "parent")),
                "children": [],
            }
    for nid, n in nodes.items():
        pid = n["parent"]
        if pid != -1 and pid in nodes:
            nodes[pid]["children"].append(nid)
    return nodes


def _find_root(nodes: Dict) -> int:
    return next(nid for nid, n in nodes.items() if n["parent"] == -1)


def _trace_branches(
    nodes: Dict, root: int
) -> Tuple[List[List[int]], List[int]]:
    branches: List[List[int]] = []
    parent_branch_idx: List[int] = []
    stack: List[Tuple[int, List[int], int]] = [(root, [root], -1)]

    while stack:
        nid, current_branch, parent_bidx = stack.pop()
        children = nodes[nid]["children"]
        if len(children) == 0:
            branches.append(current_branch)
            parent_branch_idx.append(parent_bidx)
        elif len(children) == 1:
            stack.append((children[0], current_branch + [children[0]], parent_bidx))
        else:
            my_bidx = len(branches)
            branches.append(current_branch)
            parent_branch_idx.append(parent_bidx)
            for cid in children:
                stack.append((cid, [nid, cid], my_bidx))

    return branches, parent_branch_idx


def _split_xyzr(xyzr: np.ndarray, ncomp: int) -> List[np.ndarray]:
    if len(xyzr) <= 1:
        return [xyzr] * ncomp

    xyz = xyzr[:, :3]
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]
    if total_length < 1e-10:
        return [xyzr] * ncomp

    target_dists = np.linspace(0, total_length, ncomp + 1)
    idxs = np.clip(
        np.searchsorted(cum_dists, target_dists, side="right") - 1, 0, len(xyz) - 2
    )
    local_dist = target_dists - cum_dists[idxs]
    segment_lens = np.where(dists[idxs] < 1e-14, 1e-14, dists[idxs])
    frac = (local_dist / segment_lens)[:, None]

    split_points = xyzr[idxs] + frac * (xyzr[idxs + 1] - xyzr[idxs])

    segments = []
    for i in range(ncomp):
        mask = (cum_dists > target_dists[i]) & (cum_dists < target_dists[i + 1])
        between = xyzr[mask]
        seg = np.vstack([split_points[i], *between, split_points[i + 1]])
        segments.append(seg)
    return segments


def _comp_morph_attrs(
    xyzr: np.ndarray, min_radius: float, ncomp: int
) -> Tuple[float, float, float, float, float]:
    positions = xyzr[:, :3]
    radii = xyzr[:, 3]

    if len(xyzr) > 1:
        seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        rw = np.zeros(len(seg_lengths) + 1)
        rw[1:] += seg_lengths
        rw[:-1] += seg_lengths
        rw /= rw.sum() + 1e-30
        avg_radius = float(np.sum(radii * rw))

        r_s, r_e = radii[:-1], radii[1:]
        dr = r_e - r_s
        slant = np.sqrt(dr**2 + seg_lengths**2)
        area = float(np.sum(np.pi * (r_s + r_e) * slant))
        volume = float(
            np.sum((np.pi / 3) * seg_lengths * (r_s**2 + r_s * r_e + r_e**2))
        )

        xyzr_halves = _split_xyzr(xyzr, 2)
        r_loads = []
        for half in xyzr_halves:
            p, r = half[:, :3], half[:, 3]
            if len(p) > 1:
                sl = np.linalg.norm(np.diff(p, axis=0), axis=1)
                r_loads.append(_resistive_load(sl, r))
            else:
                ln = r[0] / ncomp
                r_loads.append(ln / r[0] ** 2 / np.pi)
        r_load_in, r_load_out = r_loads[0], r_loads[1]
    else:
        avg_radius = float(radii.mean())
        area = float(4 * np.pi * radii[0] ** 2 / ncomp)
        volume = float(4 / 3 * np.pi * radii[0] ** 3 / ncomp)
        ln = radii[0] / ncomp
        r_load_in = r_load_out = float(ln / radii[0] ** 2 / np.pi)

    avg_radius = max(avg_radius, min_radius)
    return avg_radius, area, volume, r_load_in, r_load_out


def _resistive_load(lengths: np.ndarray, radii: np.ndarray) -> float:
    r_s, r_e = radii[:-1], radii[1:]
    dr = r_e - r_s
    integrals = np.empty_like(lengths)
    const = np.isclose(dr, 0)
    integrals[const] = lengths[const] / r_s[const] ** 2
    vary = ~const
    integrals[vary] = lengths[vary] / dr[vary] * (1.0 / r_s[vary] - 1.0 / r_e[vary])
    return float(np.sum(integrals) / np.pi)
