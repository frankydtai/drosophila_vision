"""Plot spot ImpR/Ca data with vs without the τ=5 pre-filter on u[t].

Current ``read_RecF_ImpR``:

    s = normalize(lowpass(u, 5))
    ImpR_i = IR_filter(s)  (+ 0.4 s for L1/L2) → normalize → × DATA_AMP × RecF

Without pre-filter:

    s = u   (still peak-normalized)
    same IR_filter / L1L2 / normalize / DATA_AMP × RecF

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_impr_prefilter.py
    ../.venv/bin/python test/plot_impr_prefilter.py --show
    ../.venv/bin/python test/plot_impr_prefilter.py --pulse-ms 50
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from figure.readout import plot_present_layout
from figure.spot import CENTER_BIN
from figure.util import TRACE_LW, TRACE_YLIM, save_figure
from neuron.params import DATA_AMP, DELTA_MS
from task.spot.data import (
    _bandpass,
    _lowpass,
    cell_list,
    normalize_data,
    read_RecF_ImpR,
)
from task.spot.input import spot_input_waveform

DEFAULT_SAVE = os.path.join(HERE, "impr_prefilter.png")
IR_HP = np.array([39.1, 28.8, 00.0, 38.1, 12.7, 31.8, 26.0, 0.00, 0.00, 29.6, 15.3, 24.9, 0.00])
IR_LP = np.array([03.8, 05.8, 05.4, 02.3, 04.2, 05.4, 02.7, 03.8, 07.7, 04.4, 01.4, 02.4, 10.7])
PREFILTER_TAU = 5.0


def _impr_cube(*, t_on: int, n_t: int, pulse_ms: float, prefilter: bool) -> np.ndarray:
    """Center-bin Ca data ``(13, n_t)`` = DATA_AMP × RecF_center × ImpR."""
    RecF, _ = read_RecF_ImpR(t_on=t_on, n_t=n_t, pulse_ms=pulse_ms)
    u = spot_input_waveform(t_on, n_t, pulse_ms)
    if prefilter:
        s = _lowpass(u, PREFILTER_TAU)
    else:
        s = np.asarray(u, dtype=np.float64).copy()
    s = s / np.max(s)

    out = np.zeros((13, n_t), dtype=np.float64)
    rf_c = RecF[:, CENTER_BIN * 5 + 2]
    for i in range(13):
        if IR_HP[i] == 0:
            impr = _lowpass(s, IR_LP[i])
        else:
            impr = _bandpass(s, IR_HP[i], IR_LP[i])
        if i < 2:
            impr = impr + 0.4 * s
        impr = normalize_data(impr)
        out[i] = DATA_AMP * rf_c[i] * impr
    return out


def _first_nz(tr: np.ndarray) -> int | None:
    abs_ = np.abs(tr)
    if not np.any(abs_ > 0):
        return None
    return int(np.argmax(abs_ > 0))


def _plot(with_pf, without_pf, *, t_on, dt_ms, pulse_ms, save, show):
    present = [str(n) for n in cell_list]
    groups, names = plot_present_layout(present)
    nrows = len(groups)
    ncols = max(len(g) for g in groups)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows), squeeze=False,
    )
    t_s = np.arange(with_pf.shape[1]) * dt_ms / 1000.0
    name_to_i = {str(n): i for i, n in enumerate(cell_list)}

    for r, group in enumerate(groups):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(group):
                ax.axis("off")
                continue
            name = str(group[c])
            i = name_to_i[name]
            w = with_pf[i]
            wo = without_pf[i]
            ax.plot(t_s, w, color="C0", lw=TRACE_LW, label="with pre-LP τ=5")
            ax.plot(t_s, wo, color="C1", lw=TRACE_LW, label="no pre-filter")
            ax.plot(t_s, w - wo, color="C3", lw=0.9, label="with − without")
            t_w = _first_nz(w)
            t_wo = _first_nz(wo)
            ax.set_title(
                f"{name}  nz@ {t_w}/{t_wo}",
                fontsize=8,
            )
            ax.set_ylim(*TRACE_YLIM)
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.axvline(t_on * dt_ms / 1000.0, color="0.85", lw=0.6, ls="--")
            if r == nrows - 1:
                ax.set_xlabel("t (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel("Ca data", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"ImpR Ca data: pre-filter τ=5 on/off  "
        f"(pulse={pulse_ms:g} ms, t_on={t_on}, Δt={dt_ms:g} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save)
    print(f"saved {save}")
    for name in names:
        i = name_to_i[name]
        print(
            f"  {name}: first_nz with={_first_nz(with_pf[i])} "
            f"without={_first_nz(without_pf[i])}"
        )
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pulse-ms", type=float, default=50.0)
    ap.add_argument("--t-on", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=151)
    ap.add_argument("--delta-ms", type=float, default=DELTA_MS)
    args = ap.parse_args()

    with_pf = _impr_cube(
        t_on=args.t_on, n_t=args.n_t, pulse_ms=args.pulse_ms, prefilter=True,
    )
    without_pf = _impr_cube(
        t_on=args.t_on, n_t=args.n_t, pulse_ms=args.pulse_ms, prefilter=False,
    )
    _plot(
        with_pf, without_pf,
        t_on=args.t_on, dt_ms=float(args.delta_ms), pulse_ms=float(args.pulse_ms),
        save=args.save, show=args.show,
    )


if __name__ == "__main__":
    main()
