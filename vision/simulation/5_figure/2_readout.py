"""Model-data readout cell selection from session + task."""

from __future__ import annotations

import numpy as np
import torch

from param_defaults import DATA_AMP, DELTA_MS
from task.spot.data import GT_CELLS, read_RecF_data, read_RecF_data_dark
from network.construction import (
    CELL_FAMILY_ROWS,
    cell_names_in_family_order,
)

PLOT_FAMILY_ROWS = [np.array(row) for row in CELL_FAMILY_ROWS]

_VALID_CONTRASTS = ("bright", "dark")


def plot_cells_in_order(present):
    """Flat cell order from :func:`network.construction.cell_names_in_family_order`."""
    return cell_names_in_family_order(present)


def _pack_for(session, task):
    if task is None:
        return session.primary_readout
    return session.pack_for(task)


def _cell_names_for_nodes(session, node_indices):
    u = node_indices
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    u = np.asarray(u, dtype=np.int64)
    C = session.backend.network
    if C is None:
        raise ValueError("_cell_names_for_nodes requires session.backend.network")
    node_cell = C.node_cell[u]
    if torch.is_tensor(node_cell):
        node_cell = node_cell.detach().cpu().numpy()
    names = list(C.cell_names)
    return [str(names[int(ti)]) for ti in node_cell]


def pack_readout_cells(session, task=None):
    """Unique cell names on pack.readout_node, pack order."""
    pack = _pack_for(session, task)
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


CONTRAST_LINESTYLE = {
    "bright": "-",
    "dark": "--",
}


def contrast_linestyle(contrast: str) -> str:
    return CONTRAST_LINESTYLE.get(str(contrast), "-")


def _mirror_data_specs_from_override(override):
    if not override:
        return []
    specs = []
    if 'mirror_fits' in override:
        for spec in override['mirror_fits']:
            specs.append((
                [str(t) for t in spec['mirror_types']],
                str(spec['mirror_fit']),
                float(spec.get('mirror_sign', -1.0)),
            ))
    elif 'mirror_fit' in override:
        spec = override['mirror_fit']
        if isinstance(spec, dict) and 'mirror_types' in spec:
            specs.append((
                [str(t) for t in spec['mirror_types']],
                str(spec['mirror_fit']),
                float(spec.get('mirror_sign', -1.0)),
            ))
    return specs


def _apply_mirror(cells, override):
    cells = dict(cells)
    for mirror_types, mirror_fit, mirror_sign in _mirror_data_specs_from_override(override):
        if mirror_fit not in cells:
            continue
        mirrored = mirror_sign * cells[mirror_fit]
        for name in mirror_types:
            cells[name] = mirrored
    return cells


def _cell_cubes(*, dark: bool, t_onset=None, n_t=None, pulse_ms=None, delta_ms: float):
    """One contrast: ``{cell: (RF_N_RADII, T)}``."""
    kw = dict(t_onset=t_onset, n_t=n_t, pulse_ms=pulse_ms, delta_ms=float(delta_ms))
    data = read_RecF_data_dark(**kw) if dark else read_RecF_data(**kw)
    cubes = data * DATA_AMP
    return {str(name): cubes[i] for i, name in enumerate(GT_CELLS)}


def fit_data_cubes(
    *, contrasts=("bright",), t_onset=None, n_t=None, pulse_ms=None, delta_ms: float = DELTA_MS,
):
    """RecF data cubes ``{contrast: {cell: (RF_N_RADII, T)}}`` (``v`` readout as-is)."""
    out = {}
    for contrast in contrasts:
        contrast = str(contrast)
        if contrast not in _VALID_CONTRASTS:
            raise ValueError(
                f"unknown contrast {contrast!r}; expected one of {_VALID_CONTRASTS}"
            )
        out[contrast] = _cell_cubes(
            dark=(contrast == "dark"),
            t_onset=t_onset, n_t=n_t, pulse_ms=pulse_ms, delta_ms=delta_ms,
        )
    return out


def spot_data_cubes(
    session,
    task=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    pulse_ms=None,
    delta_ms: float = DELTA_MS,
):
    """Spot data cubes ``{contrast: {cell: (RF_N_RADII, T)}}`` with pack mirror overrides."""
    task = task or session.primary_readout.name
    if contrasts is None:
        contrasts = (contrast_for_task(task),)
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    base = fit_data_cubes(
        contrasts=contrasts, t_onset=t_onset, n_t=n_t, pulse_ms=pulse_ms,
        delta_ms=delta_ms,
    )
    out = {}
    for contrast, cells in base.items():
        pack_name = "spot_dark" if contrast == "dark" else "spot_bright"
        ov = overrides.get(pack_name)
        if ov is None:
            ov = overrides.get(task)
        out[contrast] = _apply_mirror(cells, ov)
    return out
