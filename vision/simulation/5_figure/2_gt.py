"""Model-gt cell selection and gts from session + task."""

from __future__ import annotations

from default_params import (
    SPOT_PACK,
)

import numpy as np
import torch

from default_params import SPOT_PACK
from task.spot.gt import (
    GT_CELLS,
    RF_SIGN,
    load_gt,
    load_gt_dark,
    spot_gt_active,
)
from network.construction import active_gt_cells, cells_in_order, gt_cells_from_opts

_VALID_CONTRASTS = ("bright", "dark")


def plot_cells_in_order(active):
    """Flat cell order from :func:`network.construction.cells_in_order`."""
    return cells_in_order(active)


def _cells_for_nodes(session, node_indices):
    node_idx = node_indices
    if torch.is_tensor(node_idx):
        node_idx = node_idx.detach().cpu().numpy()
    node_idx = np.asarray(node_idx, dtype=np.int64)
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("_cells_for_nodes requires session.backend.network")
    node_cells = connectome.node_cells[node_idx]
    if torch.is_tensor(node_cells):
        node_cells = node_cells.detach().cpu().numpy()
    cells = list(connectome.cells)
    return [str(cells[int(ti)]) for ti in node_cells]


def pack_cells(session, task=None):
    """Unique cells on pack.entry_nodes, pack order."""
    pack = session.primary_pack if task is None else session.pack_for(task)
    seen = set()
    out = []
    for name in _cells_for_nodes(session, pack.entry_nodes):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def active_spot_gt_cells(session, task=None):
    """Configured spot gt cells (sti opts), not cost-pack-only.

    Falls back to :data:`GT_CELLS` when opts omit ``gt_cells``.
    """
    pack = session.primary_pack if task is None else session.pack_for(task)
    opts = dict((session.train_opts or {}).get(f"{pack.task}_sti_opts") or {})
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("active_spot_gt_cells requires session.backend.network")
    return tuple(
        active_gt_cells(
            gt_cells_from_opts(opts),
            GT_CELLS,
            connectome.cells,
            context="spot plot",
        )
    )


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


def _mirror_fit_cells(cells, pack_mirror_fit):
    cells = dict(cells)
    if not pack_mirror_fit:
        return cells
    specs = pack_mirror_fit.get("mirror_fit")
    if specs is None:
        specs = []
    elif isinstance(specs, dict):
        specs = [specs] if "mirror_types" in specs else []
    for spec in specs:
        mirror_fit = str(spec["mirror_fit"])
        if mirror_fit not in cells:
            continue
        mirrored = float(spec.get("mirror_sign", -1.0)) * cells[mirror_fit]
        for name in spec["mirror_types"]:
            cells[str(name)] = mirrored
    return cells


def fit_gts(
    *, contrasts=("bright",), t_onset=None, n_t=None, ms_sti=None,
    delta_ms, gt_amp, filter="none", spot_gt_mode="all",
):
    """``{contrast: {cell: gt}}`` with gt ``(RF_N_RADII, n_t)`` (raw gt before affine).

    Cells included only when :func:`task.spot.gt.spot_gt_active` (same as cost pack).
    """
    out = {}
    delta_ms = float(delta_ms)
    gt_amp = float(gt_amp)
    for contrast in contrasts:
        contrast = str(contrast)
        if contrast not in _VALID_CONTRASTS:
            raise ValueError(
                f"unknown contrast {contrast!r}; expected one of {_VALID_CONTRASTS}"
            )
        kw = dict(
            t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=float(delta_ms),
            filter=filter, spot_gt_mode=spot_gt_mode,
        )
        gt_stack = load_gt_dark(**kw) if contrast == "dark" else load_gt(**kw)
        scaled = gt_stack * gt_amp
        out[contrast] = {
            str(name): scaled[i]
            for i, name in enumerate(GT_CELLS)
            if spot_gt_active(spot_gt_mode, contrast, int(RF_SIGN[name]))
        }
    return out


def spot_gts(
    session,
    task=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    ms_sti=None,
    delta_ms=None,
    filter="none",
    spot_gt_mode=None,
):
    """Spot gts ``{contrast: {cell: gt}}`` with pack mirror_fit.

    ``filter`` / ``spot_gt_mode`` default from ``session.train_opts``.
    """
    task = task or session.primary_pack.task
    if contrasts is None:
        contrasts = (contrast_for_task(task),)
    pack_mirror_fits = (session.train_opts or {}).get('pack_mirror_fits') or {}
    if filter is None:
        filter = str((session.train_opts or {}).get("filter", "none"))
    else:
        filter = str(filter)
    if spot_gt_mode is None:
        spot_gt_mode = str((session.train_opts or {}).get("spot_gt_mode", SPOT_PACK['spot_gt_mode']))
    else:
        spot_gt_mode = str(spot_gt_mode)
    base = fit_gts(
        contrasts=contrasts, t_onset=t_onset, n_t=n_t, ms_sti=ms_sti,
        delta_ms=session.delta_ms if delta_ms is None else delta_ms,
        gt_amp=session.gt_amp,
        filter=filter,
        spot_gt_mode=spot_gt_mode,
    )
    out = {}
    for contrast, cells in base.items():
        task = "spot_dark" if contrast == "dark" else "spot_bright"
        pack_mirror_fit = pack_mirror_fits.get(task)
        out[contrast] = _mirror_fit_cells(cells, pack_mirror_fit)
    return out
