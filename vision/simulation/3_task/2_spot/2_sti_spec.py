# -*- coding: utf-8 -*-
"""Spot sti spec: ``a_sti_radius`` drive (timing from :mod:`task.spread.sti_spec`)."""
from __future__ import annotations

from typing import Sequence

import torch

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
    sim_dtype,
    device,
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
                a_sti_radius_idx = radius_idx[radius]
                for du, dv in build_hex.shell_hexes(radius):
                    for node in connectome.sti_nodes_at_uv(
                        int(sti_hex_u) + int(du), int(sti_hex_v) + int(dv),
                    ):
                        sti_bs.append(int(b))
                        sti_nodes.append(int(node))
                        a_sti_radius_idxs.append(int(a_sti_radius_idx))
    mask = sti_mask(t_onset, n_t, ms_sti, delta_ms=delta_ms)
    n_b = len(spot_bs)
    i_sti = float(i_sti)
    network_sti_nodes = torch.as_tensor(connectome.sti_nodes, dtype=torch.long, device=device)
    i_sti_pulse = torch.as_tensor(
        (i_sti - float(i_baseline)) * mask, dtype=sim_dtype, device=device,
    )
    i_sti = torch.zeros((n_b, n_t, connectome.n_node), dtype=sim_dtype, device=device)
    if len(network_sti_nodes):
        i_sti[:, :, network_sti_nodes] = float(i_baseline)
    for b, node in center_nodes:
        i_sti[b, :, node] = i_sti[b, :, node] + i_sti_pulse
    sti_bs_t = torch.tensor(sti_bs, dtype=torch.long, device=device)
    sti_nodes_t = torch.tensor(sti_nodes, dtype=torch.long, device=device)
    a_sti_radius_idxs_t = torch.tensor(a_sti_radius_idxs, dtype=torch.long, device=device)
    return i_sti, i_sti_pulse, sti_bs_t, sti_nodes_t, a_sti_radius_idxs_t
