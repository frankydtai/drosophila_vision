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
from network.construction import active_gt_cells, gt_cells_from_opts

_VALID_CONTRASTS = ("bright", "dark")


def cells_from_nodes(session, node_indices):
    node_idx = node_indices
    if torch.is_tensor(node_idx):
        node_idx = node_idx.detach().cpu().numpy()
    node_idx = np.asarray(node_idx, dtype=np.int64)
    connectome = session.backend.network
    if connectome is None:
        raise ValueError("cells_from_nodes requires session.backend.network")
    node_cells = connectome.node_cells[node_idx]
    if torch.is_tensor(node_cells):
        node_cells = node_cells.detach().cpu().numpy()
    cells = list(connectome.cells)
    return [str(cells[int(ti)]) for ti in node_cells]


def pack_cells(session, task=None):
    """Unique cells on pack.entry_nodes, pack order."""
    pack = session.primary_pack if task is None else session.pack_from_task(task)
    seen = set()
    out = []
    for name in cells_from_nodes(session, pack.entry_nodes):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def active_spot_gt_cells(session, task=None):
    """Configured spot gt cells (sti opts), not cost-pack-only.

    Falls back to :data:`GT_CELLS` when opts omit ``gt_cells``.
    """
    pack = session.primary_pack if task is None else session.pack_from_task(task)
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


def contrast_from_task(task) -> str:
    """``bright`` / ``dark`` from a spot pack name."""
    return "dark" if str(task) == "spot_dark" else "bright"


def contrast_order(contrasts) -> tuple[str, ...]:
    """Stable plot order: bright, dark, then any extras."""
    preferred = ("bright", "dark")
    keys = [str(contrast) for contrast in contrasts]
    return tuple(contrast for contrast in preferred if contrast in keys) + tuple(
        contrast for contrast in keys if contrast not in preferred
    )


def contrast_linestyle(contrast: str) -> str:
    return {"bright": "-", "dark": "--"}.get(str(contrast), "-")


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
    """Spot gts ``{contrast: {cell: gt}}``.

    ``filter`` / ``spot_gt_mode`` default from ``session.train_opts``.
    """
    task = task or session.primary_pack.task
    if contrasts is None:
        contrasts = (contrast_from_task(task),)
    if filter is None:
        filter = str((session.train_opts or {}).get("filter", "none"))
    else:
        filter = str(filter)
    if spot_gt_mode is None:
        spot_gt_mode = str((session.train_opts or {}).get("spot_gt_mode", SPOT_PACK['spot_gt_mode']))
    else:
        spot_gt_mode = str(spot_gt_mode)
    delta_ms = float(session.delta_ms if delta_ms is None else delta_ms)
    gt_amp = float(session.gt_amp)
    out = {}
    for contrast in contrasts:
        contrast = str(contrast)
        if contrast not in _VALID_CONTRASTS:
            raise ValueError(
                f"unknown contrast {contrast!r}; expected one of {_VALID_CONTRASTS}"
            )
        kw = dict(
            t_onset=t_onset, n_t=n_t, ms_sti=ms_sti, delta_ms=delta_ms,
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
