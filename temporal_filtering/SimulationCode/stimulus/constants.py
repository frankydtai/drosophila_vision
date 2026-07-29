# -*- coding: utf-8 -*-
"""Fit-cell vocabulary shared by the stimulus paradigms.

Split out of the old ``Medulla_Library``. Drive currents (``I_BASELINE`` etc.)
and the target amplitude scale live in the top-level ``config`` because the
network signal builder needs them too; here we keep only the 13-cell fit list
that is a target/readout concept (stimulus + plot only).
"""
from __future__ import annotations

import numpy as np

cell_list = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)
N_FIT_CELLS = len(cell_list)

LEAK_DEPOL_TYPES = ['L1', 'L2', 'L3']  # [] -> all -50 mV


def fit_list_index(name: str) -> int:
    matches = np.where(cell_list == name)[0]
    if len(matches) != 1:
        raise KeyError(f"fit cell {name!r} not in cell_list")
    return int(matches[0])
