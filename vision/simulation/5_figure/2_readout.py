"""Model-gt readout cell selection from session + task."""

from __future__ import annotations

import numpy as np
import torch

from param_defaults import GT_CA_AMP, GT_V_AMP, DELTA_MS
from task.spot.gt import GT_CELLS, read_RecF_gt, read_RecF_gt_dark
from network.construction import cell_names_in_order

_VALID_CONTRASTS = ("bright", "dark")


def plot_cells_in_order(present):
    """Flat cell order from :func:`network.construction.cell_names_in_order`."""
    return cell_names_in_order(present)


def _cell_names_for_nodes(session, node_indices):
    node_index = node_indices
    if torch.is_tensor(node_index):
        node_index = node_index.detach().cpu().numpy()
    node_index = np.asarray(node_index, dtype=np.int64)
    C = session.backend.network
    if C is None:
        raise ValueError("_cell_names_for_nodes requires session.backend.network")
    node_cell = C.node_cell[node_index]
    if torch.is_tensor(node_cell):
        node_cell = node_cell.detach().cpu().numpy()
    names = list(C.cell_names)
    return [str(names[int(ti)]) for ti in node_cell]


def pack_readout_cells(session, task=None):
    """Unique cell names on pack.readout_node, pack order."""
    pack = session.primary_readout if task is None else session.pack_for(task)
    seen = set()
    out = []
    for name in _cell_names_for_nodes(session, pack.readout_node):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def contrast_for_task(task) -> str:
    """``bright`` / ``dark`` from a spot pack name."""
    return "dark" if str(task) == "spot_dark" else "bright"


def contrast_order(contrasts) -> tuple[str, ...]:
    """Stable plot order: bright, dark, then any extras."""
    preferred = ("bright", "dark")
    keys = [str(c) for c in contrasts]
    return tuple(c for c in preferred if c in keys) + tuple(
        c for c in keys if c not in preferred
    )


def contrast_linestyle(contrast: str) -> str:
    return {"bright": "-", "dark": "--"}.get(str(contrast), "-")


def _apply_mirror(cells, override):
    cells = dict(cells)
    if not override:
        return cells
    if "mirror_fits" in override:
        specs = override["mirror_fits"]
    elif "mirror_fit" in override:
        spec = override["mirror_fit"]
        specs = [spec] if isinstance(spec, dict) and "mirror_types" in spec else []
    else:
        specs = []
    for spec in specs:
        mirror_fit = str(spec["mirror_fit"])
        if mirror_fit not in cells:
            continue
        mirrored = float(spec.get("mirror_sign", -1.0)) * cells[mirror_fit]
        for name in spec["mirror_types"]:
            cells[str(name)] = mirrored
    return cells


def fit_gt_cubes(
    *, contrasts=("bright",), t_onset=None, n_t=None, ms_spot=None,
    delta_ms: float = DELTA_MS, filter="none",
):
    """RecF gt cubes ``{contrast: {cell: (RF_N_RADII, T)}}`` (raw gt before affine).

    ``filter==\"ca\"`` → Arenz digitized ImpR (same as training ``build_spot_gt``).
    """
    out = {}
    for contrast in contrasts:
        contrast = str(contrast)
        if contrast not in _VALID_CONTRASTS:
            raise ValueError(
                f"unknown contrast {contrast!r}; expected one of {_VALID_CONTRASTS}"
            )
        kw = dict(
            t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=float(delta_ms),
            filter=filter,
        )
        gt = read_RecF_gt_dark(**kw) if contrast == "dark" else read_RecF_gt(**kw)
        gt_amp = GT_CA_AMP if filter == "ca" else GT_V_AMP
        cubes = gt * gt_amp
        out[contrast] = {str(name): cubes[i] for i, name in enumerate(GT_CELLS)}
    return out


def spot_gt_cubes(
    session,
    task=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    ms_spot=None,
    delta_ms: float = DELTA_MS,
    filter=None,
):
    """Spot gt cubes ``{contrast: {cell: (RF_N_RADII, T)}}`` with pack mirror overrides.

    ``filter`` defaults to ``session.train_opts['filter']`` (training GT kind).
    """
    task = task or session.primary_readout.name
    if contrasts is None:
        contrasts = (contrast_for_task(task),)
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    if filter is None:
        filter = str((session.train_opts or {}).get("filter", "none"))
    else:
        filter = str(filter)
    base = fit_gt_cubes(
        contrasts=contrasts, t_onset=t_onset, n_t=n_t, ms_spot=ms_spot,
        delta_ms=delta_ms, filter=filter,
    )
    out = {}
    for contrast, cells in base.items():
        pack_name = "spot_dark" if contrast == "dark" else "spot_bright"
        ov = overrides.get(pack_name)
        if ov is None:
            ov = overrides.get(task)
        out[contrast] = _apply_mirror(cells, ov)
    return out
