from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


SIMULATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMULATION_ROOT not in sys.path:
    sys.path.insert(0, SIMULATION_ROOT)

import import_bootstrap  # noqa: E402,F401
from figure.sbar import resolve_sbar_gts  # noqa: E402


def _session(sbar_gt_w: int):
    pack = SimpleNamespace(
        task="sbar",
        contrast="bright",
        t_onset=0,
        i_sti=torch.zeros((1, 161, 1)),
        gts=torch.zeros((1, 161)),
        entry_part_keys=("sbar_bright_T4a_mid+0",),
    )
    return SimpleNamespace(
        primary_pack=pack,
        train_opts={
            "sbar_sti_opts": {
                "ms_response": 160,
                "ms_sti": 160,
                "ms_post": 0,
                "delta_ms": 1.0,
                "sbar_gt_w": sbar_gt_w,
            },
        },
        connectome=SimpleNamespace(cells=("T4a",)),
    )


def test_figure_resolves_configured_sbar_gt_width():
    w1, _ = resolve_sbar_gts({"bright": _session(1)})
    w2, _ = resolve_sbar_gts({"bright": _session(2)})

    w1_trace = w1["bright"][("T4a", 0.0)]
    w2_trace = w2["bright"][("T4a", 0.0)]
    assert not np.allclose(w1_trace, w2_trace, equal_nan=True)
