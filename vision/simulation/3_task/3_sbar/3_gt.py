# -*- coding: utf-8 -*-
"""Static-bar GT numbers: Gruntman Fig.2 digitized width-1 traces."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuron.borst import t_from_ms


def load_gt(
    *,
    t_onset,
    ms_response,
    ms_sti,
    delta_ms: float,
    ms_post=0.0,
):
    """Load width-1 digitized traces keyed by CSV ``trace_id``."""
    ms_response = float(ms_response)
    if ms_sti is not None:
        ms_response = max(ms_response, float(ms_sti))
    t_axis = (
        np.arange(
            int(
                int(t_onset)
                + t_from_ms(ms_response, delta_ms=float(delta_ms))
                + t_from_ms(float(ms_post), delta_ms=float(delta_ms))
                + 1
            ),
            dtype=np.float64,
        )
        * float(delta_ms)
    )
    gts = {}
    for _, grouped in pd.read_csv(
        Path(__file__).resolve().parents[4]
        / "figure_digitization"
        / "gruntman21"
        / "2ax2bc_digitized.csv"
    ).query("target_width_led == 1").groupby("trace_id"):
        gt = np.interp(
            t_axis,
            grouped["time_ms"].to_numpy(dtype=np.float64),
            grouped["vm_mv"].to_numpy(dtype=np.float64),
            left=float(grouped["vm_mv"].iloc[0]),
            right=float(grouped["vm_mv"].iloc[-1]),
        )
        gts[str(grouped["trace_id"].iloc[0])] = gt
    return gts
