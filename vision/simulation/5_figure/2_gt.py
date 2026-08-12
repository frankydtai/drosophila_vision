"""Model-gt cell selection and gt rts from session + task."""

from __future__ import annotations

from default_params import (
    SPOT_PACK_DEFAULT,
)

import numpy as np
import torch

from default_params import SPOT_PACK_DEFAULT['spot_gt_mode']
from task.spot.gt import (
    GT_CELLS,
    RF_SIGN,
    load_RecF_gt,
    load_RecF_gt_dark,
    spot_gt_active,
)
from network.construction import cells_in_order

_VALID_CONTRASTS = ("bright", "dark")


def plot_cells_in_order(present):
    """Flat cell order from :func:`network.construction.cells_in_order`."""
    return cells_in_order(present)


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
    names = list(connectome.cells)
    return [str(names[int(ti)]) for ti in node_cells]


def pack_cells(session, task=None):
    """Unique cell names on pack.entry_nodes, pack order."""
    pack = session.primary_pack if task is None else session.pack_for(task)
    seen = set()
    out = []
    for name in _cells_for_nodes(session, pack.entry_nodes):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def present_spot_gt_cells(session, task=None):
    """Configured spot gt cells (sti opts), not cost-pack-only.

    Falls back to :func:`pack_cells` when opts omit ``gt_cells``.
    """
    pack = session.primary_pack if task is None else session.pack_for(task)
    opts = dict((session.train_opts or {}).get(f"{pack.name}_sti_opts") or {})
    raw = opts.get("gt_cells")
    if raw:
        return tuple(str(name) for name in raw)
    return pack_cells(session, pack.name)


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


def fit_gt_rts(
    *, contrasts=("bright",), t_onset=None, n_t=None, ms_spot=None,
    delta_ms, gt_amp, filter="none", spot_gt_mode="all",
):
    """RecF gt rts ``{contrast: {cell: (RF_N_RADII, T)}}`` (raw gt before affine).

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
            t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=float(delta_ms),
            filter=filter, spot_gt_mode=spot_gt_mode,
        )
        gt = load_RecF_gt_dark(**kw) if contrast == "dark" else load_RecF_gt(**kw)
        rts = gt * gt_amp
        out[contrast] = {
            str(name): rts[i]
            for i, name in enumerate(GT_CELLS)
            if spot_gt_active(spot_gt_mode, contrast, int(RF_SIGN[i]))
        }
    return out


def spot_gt_rts(
    session,
    task=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    ms_spot=None,
    delta_ms=None,
    filter="none",
    spot_gt_mode=None,
):
    """Spot gt rts ``{contrast: {cell: (RF_N_RADII, T)}}`` with pack mirror overrides.

    ``filter`` / ``spot_gt_mode`` default from ``session.train_opts``.
    """
    task = task or session.primary_pack.name
    if contrasts is None:
        contrasts = (contrast_for_task(task),)
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    if filter is None:
        filter = str((session.train_opts or {}).get("filter", "none"))
    else:
        filter = str(filter)
    if spot_gt_mode is None:
        spot_gt_mode = str((session.train_opts or {}).get("spot_gt_mode", SPOT_PACK_DEFAULT['spot_gt_mode']))
    else:
        spot_gt_mode = str(spot_gt_mode)
    base = fit_gt_rts(
        contrasts=contrasts, t_onset=t_onset, n_t=n_t, ms_spot=ms_spot,
        delta_ms=session.delta_ms if delta_ms is None else delta_ms,
        gt_amp=session.gt_amp,
        filter=filter,
        spot_gt_mode=spot_gt_mode,
    )
    out = {}
    for contrast, cells in base.items():
        pack_name = "spot_dark" if contrast == "dark" else "spot_bright"
        ov = overrides.get(pack_name)
        if ov is None:
            ov = overrides.get(task)
        out[contrast] = _apply_mirror(cells, ov)
    return out
