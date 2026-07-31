"""Model-data readout cell selection from session + target."""

from __future__ import annotations

import numpy as np
import torch

from training.defaults import DATA_AMP, DELTA_MS
from task.spot.data import cell_list, read_RecF_data, read_RecF_data_dark
from network.construction import (
    TYPE_FAMILY_ROWS,
    type_family_row_groups,
    type_names_in_family_order,
)

PLOT_FAMILY_ROWS = [np.array(row) for row in TYPE_FAMILY_ROWS]

_VALID_CONTRASTS = ("bright", "dark")


def plot_row_groups(present):
    """Family row groups for plots; skip absent types and empty rows."""
    return [np.array(row) for row in type_family_row_groups(present)]


def plot_present_layout(present):
    """Return ``(groups, names)`` in canonical family order."""
    groups = plot_row_groups(present)
    names = [str(n) for row in groups for n in row]
    return groups, names


def plot_types_in_order(present):
    """Flat cell-type order from :func:`plot_present_layout`."""
    return type_names_in_family_order(present)


def _pack_for(session, target):
    if target is None:
        return session.primary_pack
    return session.pack_for(target)


def _type_names_for_units(session, unit_indices):
    u = unit_indices
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    u = np.asarray(u, dtype=np.int64)
    C = session.backend.network
    if C is None:
        raise ValueError("_type_names_for_units requires session.backend.network")
    node_type = C.node_type[u]
    if torch.is_tensor(node_type):
        node_type = node_type.detach().cpu().numpy()
    names = list(C.type_names)
    return [str(names[int(ti)]) for ti in node_type]


def pack_readout_types(session, target=None):
    """Unique cell-type names on pack.readout_unit, pack order."""
    pack = _pack_for(session, target)
    seen = set()
    out = []
    for name in _type_names_for_units(session, pack.readout_unit):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def contrast_for_target(target) -> str:
    """``bright`` / ``dark`` from a spot pack name."""
    return "dark" if str(target) == "spot_dark" else "bright"


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
    """One contrast: ``{cell: (9, T)}``."""
    kw = dict(t_onset=t_onset, n_t=n_t, pulse_ms=pulse_ms, delta_ms=float(delta_ms))
    data = read_RecF_data_dark(**kw) if dark else read_RecF_data(**kw)
    cubes = data * DATA_AMP
    return {str(name): cubes[i] for i, name in enumerate(cell_list)}


def fit_data_cubes(
    *, contrasts=("bright",), t_onset=None, n_t=None, pulse_ms=None, delta_ms: float = DELTA_MS,
):
    """RecF data cubes ``{contrast: {cell: (9, T)}}`` (``v`` target as-is)."""
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
    target=None,
    *,
    contrasts=None,
    t_onset=None,
    n_t=None,
    pulse_ms=None,
    delta_ms: float = DELTA_MS,
):
    """Spot data cubes ``{contrast: {cell: (9, T)}}`` with pack mirror overrides."""
    target = target or session.primary_pack.name
    if contrasts is None:
        contrasts = (contrast_for_target(target),)
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
            ov = overrides.get(target)
        out[contrast] = _apply_mirror(cells, ov)
    return out
