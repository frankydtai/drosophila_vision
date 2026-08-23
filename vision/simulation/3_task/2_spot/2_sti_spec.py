# -*- coding: utf-8 -*-
"""Spot sti spec: ``a_sti_radius`` drive (timing from :mod:`task.spread.sti_spec`)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from network import path  # noqa: F401 -- FAFBv783 on sys.path
import build_hex
from task.spot.sti_geo import SpotB
from task.spread.sti_spec import sti_mask

__all__ = (
    "build_spot_a_sti_radius_drive",
    "sti_mask",
)


def build_spot_a_sti_radius_drive(
    connectome,
    spot_bs: Sequence[SpotB],
    *,
    a_sti_radii,
    t_onset: int,
    n_t: int,
    ms_sti,
    delta_ms: float,
    i_baseline: float,
    i_sti: float,
):
    radii = tuple(int(radius) for radius in a_sti_radii)
    if any(radius == 0 for radius in radii):
        raise ValueError("a_sti_radii must omit center radius=0 (baked into i_sti @1)")
    radius_idx = dict(zip(radii, range(len(radii))))
    sti_bs: list[int] = []
    sti_nodes: list[int] = []
    a_sti_radius_idxs: list[int] = []
    center_nodes: list[tuple[int, int]] = []
    for b, spot_b in enumerate(spot_bs):
        for sti_hex_u, sti_hex_v in spot_b.sti_uv:
            for node in connectome.sti_nodes_at_uv(int(sti_hex_u), int(sti_hex_v)):
                center_nodes.append((int(b), int(node)))
            for radius in radii:
                for du, dv in build_hex.shell_hexes(radius):
                    for node in connectome.sti_nodes_at_uv(
                        int(sti_hex_u) + int(du), int(sti_hex_v) + int(dv),
                    ):
                        sti_bs.append(int(b))
                        sti_nodes.append(int(node))
                        a_sti_radius_idxs.append(int(radius_idx[radius]))
    i_sti_pulse = (float(i_sti) - float(i_baseline)) * sti_mask(
        t_onset, n_t, ms_sti, delta_ms=delta_ms,
    )
    i_sti = np.zeros((len(spot_bs), n_t, connectome.n_node), dtype=np.float64)
    network_sti_nodes = np.asarray(connectome.sti_nodes, dtype=np.int64)
    if len(network_sti_nodes):
        i_sti[:, :, network_sti_nodes] = float(i_baseline)
    for b, node in center_nodes:
        i_sti[b, :, node] = i_sti[b, :, node] + i_sti_pulse
    return (
        i_sti,
        i_sti_pulse,
        np.asarray(sti_bs, dtype=np.int64),
        np.asarray(sti_nodes, dtype=np.int64),
        np.asarray(a_sti_radius_idxs, dtype=np.int64),
    )
