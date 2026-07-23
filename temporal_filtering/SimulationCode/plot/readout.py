"""Model-data readout cell selection from session + target."""

from __future__ import annotations

import numpy as np
import torch

import Medulla_Library as ml

PLOT_FAMILY_ROWS = [
    np.array(['R1-6', 'R7', 'R8']),
    np.array(['L1', 'L2', 'L3', 'L4', 'L5']),
    np.array(['Mi1', 'Mi4', 'Mi9']),
    np.array(['T1', 'T2', 'T2a', 'T3']),
    np.array(['T4a', 'T4b', 'T4c', 'T4d']),
    np.array(['T5a', 'T5b', 'T5c', 'T5d']),
    np.array(['Tm1', 'Tm2', 'Tm20', 'Tm21', 'Tm3', 'Tm4', 'Tm9']),
    np.array(['C2', 'C3']),
]


def plot_row_groups(present):
    """Family row groups for plots; skip absent types and empty rows."""
    present_list = [str(t) for t in present]
    present_set = set(present_list)
    rows = []
    used = set()
    for row in PLOT_FAMILY_ROWS:
        filtered = [str(t) for t in row if str(t) in present_set]
        if filtered:
            rows.append(np.array(filtered))
            used.update(filtered)
    for name in present_list:
        if name not in used:
            rows.append(np.array([name]))
    return rows


def plot_present_layout(present):
    """Return ``(groups, names)`` in canonical family order."""
    groups = plot_row_groups(present)
    names = [str(n) for row in groups for n in row]
    return groups, names


def plot_types_in_order(present):
    """Flat cell-type order from :func:`plot_present_layout`."""
    return plot_present_layout(present)[1]


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


def _mirror_ref_specs_from_override(override):
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


def fit_ref_cubes(dark=False):
    """RecF reference cubes for the 13 fit cell types."""
    data = ml.read_RecF_data_dark() if dark else ml.read_RecF_data()
    ref = data * ml.DATA_AMP
    return {str(name): ref[i] for i, name in enumerate(ml.cell_list)}


def spot_ref_cubes(session, target=None, dark=False):
    """Spot model-data reference cubes from ``read_RecF_data`` (shape ``(9, T)``)."""
    target = target or session.primary_pack.name
    ref = dict(fit_ref_cubes(dark=dark))
    overrides = (session.train_opts or {}).get('pack_overrides') or {}
    for mirror_types, mirror_fit, mirror_sign in _mirror_ref_specs_from_override(
        overrides.get(target),
    ):
        if mirror_fit not in ref:
            continue
        mirrored = mirror_sign * ref[mirror_fit]
        for name in mirror_types:
            ref[name] = mirrored
    return ref
