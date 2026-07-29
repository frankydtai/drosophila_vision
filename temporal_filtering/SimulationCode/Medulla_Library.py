# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 08:43:49 2018

@author: aborst
"""

import numpy as np
import blindschleiche_py3 as bs

from training_config import (
    RESPONSE_DURATION_MS,
    ms_to_steps,
)

I_BASELINE = 20.0  # pA photoreceptor current before stimulus onset
I_BRIGHT = 40.0    # pA photoreceptor current at bright / on-step peak
I_DARK = 0.0       # pA photoreceptor current at full dark-bar coverage
DATA_AMP = 20.0         # pA scale on ImpR target traces (fit cells)

cell_list = np.array(
    ['L1', 'L2', 'L3', 'L4', 'L5', 'Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
)
N_FIT_CELLS = len(cell_list)

LEAK_DEPOL_TYPES = ['L1', 'L2', 'L3']  # [] → all -50 mV


def fit_list_index(name: str) -> int:
    matches = np.where(cell_list == name)[0]
    if len(matches) != 1:
        raise KeyError(f"fit cell {name!r} not in cell_list")
    return int(matches[0])


def normalize_data(x):

    x = x - x[0]

    mymax = np.nanmax(x)
    mymin = np.nanmin(x)

    if np.abs(mymax) > np.abs(mymin):
        absmax = np.abs(mymax)
    else:
        absmax = np.abs(mymin)

    result = x / absmax

    if mymax == mymin:
        result = x * 0.0

    return result


def read_RecF_ImpR(*, t_on=None, maxtime=None):
    """Return ``(RecF_data, ImpR_data)`` for the 13 fit cell types.

    Shapes: ``RecF_data`` ``(13, 45)``; ``ImpR_data`` ``(13, maxtime)``.
    Time axis: :mod:`training_config`.

    Split out of :func:`read_RecF_data` for hex spot targets that sample RecF at
    non-integer column distances (e.g. ``r=sqrt(3)``).
    """
    if t_on is None or maxtime is None:
        raise ValueError("read_RecF_ImpR requires t_on and maxtime")
    t_on = int(t_on)
    maxtime = int(maxtime)

    RF_center_width = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
    RF_surrnd_width = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
    RF_surrnd_weight = np.array(
        [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
    ) * 5.0
    RF_sign = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    RecF_data = np.zeros((13, 45))

    for i in range(13):

        center = bs.Gauss1D(RF_center_width[i], 44)
        surrnd = bs.Gauss1D(RF_surrnd_width[i], 44)

        RecF_data[i] = (center - RF_surrnd_weight[i] * surrnd) * RF_sign[i]
        RecF_data[i] = normalize_data(RecF_data[i])

    # hp and lp time constants * 10 ms

    IR_hp = np.array([39.1, 28.8, 00.0, 38.1, 12.7, 31.8, 26.0, 0.00, 0.00, 29.6, 15.3, 24.9, 0.00])
    IR_lp = np.array([03.8, 05.8, 05.4, 02.3, 04.2, 05.4, 02.7, 03.8, 07.7, 04.4, 01.4, 02.4, 10.7])

    signal = np.zeros(maxtime)
    signal[t_on:maxtime] = 1.0
    signal = bs.lowpass(signal, 5)
    signal = signal / np.max(signal)

    ImpR_data = np.zeros((13, maxtime))

    for i in range(13):

        if IR_hp[i] == 0:

            ImpR_data[i] = bs.lowpass(signal, IR_lp[i])

        else:

            ImpR_data[i] = bs.bandpass(signal, IR_hp[i], IR_lp[i])

        # L1 and L2

        if i < 2:

            ImpR_data[i] = ImpR_data[i] + 0.4 * signal

        ImpR_data[i] = normalize_data(ImpR_data[i])

    return RecF_data, ImpR_data


def read_RecF_data(*, t_on=None, maxtime=None):
    """Spatial×temporal spot cube ``(13, 9, maxtime)``.

    ``data[i, j, :] = RecF_data[i, 5*j+2] * ImpR_data[i]``. Time axis:
    :mod:`training_config`. Spatial ``j=0…8`` (centre ``j=4``).
    """
    RecF_data, ImpR_data = read_RecF_ImpR(t_on=t_on, maxtime=maxtime)
    mt = ImpR_data.shape[1]

    data = np.zeros((13, 9, mt))

    for i in range(13):
        for j in range(9):
            data[i, j] = RecF_data[i, j * 5 + 2] * ImpR_data[i]

    return data


def read_RecF_data_dark(*, t_on=None, maxtime=None):
    """Dark spot spatial×temporal cube: negated bright ``read_RecF_data()``."""
    return -read_RecF_data(t_on=t_on, maxtime=maxtime)
