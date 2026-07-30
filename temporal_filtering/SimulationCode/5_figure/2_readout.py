"""Model-data readout cell selection from session + target."""

from __future__ import annotations

import numpy as np
import torch

from neuron.params import DATA_AMP
from neuron import ca_to_v_delta
from task.spot.data import cell_list, read_RecF_data, read_RecF_data_dark
from network.construction import (
    TYPE_FAMILY_ROWS,
    type_family_row_groups,
    type_names_in_family_order,
)

PLOT_FAMILY_ROWS = [np.array(row) for row in TYPE_FAMILY_ROWS]


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


def fit_ref_cubes(dark=False, *, t_on=None, n_t=None, pulse_ms=None, v_delta=False):
    """RecF reference cubes for the 13 fit cell types.

    ``v_delta`` inverts the Ca low-pass (``ca_to_v_delta``) so the gray data
    matches a model ``'v'`` readout (#5), using the same filter as ``--filter v``.
    """
    kw = dict(t_on=t_on, n_t=n_t, pulse_ms=pulse_ms)
    data = read_RecF_data_dark(**kw) if dark else read_RecF_data(**kw)
    ref = data * DATA_AMP
    if v_delta:
        ref = ca_to_v_delta(ref, t_on=int(t_on or 0))
    return {str(name): ref[i] for i, name in enumerate(cell_list)}


def spot_ref_cubes(
    session,
    target=None,
    dark=False,
    *,
    t_on=None,
    n_t=None,
    pulse_ms=None,
    v_delta=False,
):
    """Spot model-data reference cubes from ``read_RecF_data`` (shape ``(9, T)``)."""
    target = target or session.primary_pack.name
    ref = dict(
        fit_ref_cubes(
            dark=dark, t_on=t_on, n_t=n_t, pulse_ms=pulse_ms, v_delta=v_delta,
        )
    )
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
