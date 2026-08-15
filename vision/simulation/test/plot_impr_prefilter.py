"""Plot spot ImpR/Ca gt with vs without a 350 ms drive low-pass.

Test drive levels (ratio like network 20/40 pA): baseline 0.5, peak 1.0.

    s = LP(u, τ=350 ms) / max   (or s = u / max without LP)
    ImpR_i = IR_filter(s)  (+ 0.4 s for L1/L2) → normalize → × GT_AMP × RecF

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_impr_prefilter.py
    ../.venv/bin/python test/plot_impr_prefilter.py --show
    ../.venv/bin/python test/plot_impr_prefilter.py --ms-pulse 50
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

from network.construction import cell_order_rows, cell_names_in_order
from figure.spot import CENTER_BIN
from figure.panel import TRACE_LW, TRACE_YLIM, save_figure
from neuron.param import DELTA_MS, t_from_ms
from const_default import GT_AMP
from task.spot.gt import (
    _bandpass,
    _lowpass,
    GT_CELLS,
    normalize_gt,
    read_RecF_ImpR,
)
from task.spot.input import spot_input_waveform

DEFAULT_SAVE = os.path.join(HERE, "impr_prefilter.png")
IR_HP = np.array([39.1, 28.8, 00.0, 38.1, 12.7, 31.8, 26.0, 0.00, 0.00, 29.6, 15.3, 24.9, 0.00])
IR_LP = np.array([03.8, 05.8, 05.4, 02.3, 04.2, 05.4, 02.7, 03.8, 07.7, 04.4, 01.4, 02.4, 10.7])
PREFILTER_MS = 350.0
U_BASELINE = 0.5
U_PEAK = 1.0


def _drive_u_s(
    *, t_on: int, n_t: int, ms_pulse: float, dt_ms: float, prefilter_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """PR drive ``u`` and peak-normalized LP(u, prefilter_ms) ``s``."""
    gate = np.asarray(spot_input_waveform(t_on, n_t, ms_pulse), dtype=np.float64)
    u = U_BASELINE + (U_PEAK - U_BASELINE) * gate
    tau = float(t_from_ms(prefilter_ms, delta_ms=dt_ms))
    s = _lowpass(u, tau)
    s = s / np.max(s)
    return u, s


def _impr_rt(
    *, t_on: int, n_t: int, ms_pulse: float, dt_ms: float, prefilter_ms: float | None,
) -> np.ndarray:
    """Center-bin Ca gt ``(13, n_t)`` = GT_AMP × RecF_center × ImpR."""
    RecF, _ = read_RecF_ImpR(t_on=t_on, n_t=n_t, ms_pulse=ms_pulse)
    u, s_lp = _drive_u_s(
        t_on=t_on, n_t=n_t, ms_pulse=ms_pulse, dt_ms=dt_ms, prefilter_ms=prefilter_ms or PREFILTER_MS,
    )
    if prefilter_ms is not None:
        s = s_lp
    else:
        s = u / np.max(u)

    out = np.zeros((13, n_t), dtype=np.float64)
    rf_c = RecF[:, CENTER_BIN * 5 + 2]
    for i in range(13):
        if IR_HP[i] == 0:
            impr = _lowpass(s, IR_LP[i])
        else:
            impr = _bandpass(s, IR_HP[i], IR_LP[i])
        if i < 2:
            impr = impr + 0.4 * s
        impr = normalize_gt(impr)
        out[i] = GT_AMP * rf_c[i] * impr
    return out


def _first_nz(tr: np.ndarray) -> int | None:
    abs_ = np.abs(tr)
    if not np.any(abs_ > 0):
        return None
    return int(np.argmax(abs_ > 0))


def _first_change(tr: np.ndarray) -> int | None:
    """First index differing from ``tr[0]`` (for nonzero baseline drives)."""
    d = np.abs(tr - tr[0])
    if not np.any(d > 0):
        return None
    return int(np.argmax(d > 0))


def _plot(
    with_lp, without_lp, u, s_lp, *, t_on, dt_ms, ms_pulse, prefilter_ms, save, show,
):
    active = [str(n) for n in GT_CELLS]
    groups = [np.array(row) for row in cell_order_rows(active)]
    names = cell_names_in_order(active)
    nrows = len(groups)
    ncols = max(len(cell_group) for cell_group in groups)
    fig = plt.figure(figsize=(2.2 * ncols, 1.6 + 2.0 * nrows))
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1.1] + [1.0] * nrows)
    ax_pr = fig.add_subplot(gs[0, :])
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for col in range(ncols):
            axes[row][col] = fig.add_subplot(gs[row + 1, col])

    t_s = np.arange(with_lp.shape[1]) * dt_ms / 1000.0
    idx_from_cell = {str(n): i for i, n in enumerate(GT_CELLS)}
    tau_t = t_from_ms(prefilter_ms, delta_ms=dt_ms)

    ax_pr.plot(t_s, u, color="0.25", lw=TRACE_LW, label=f"PR drive u ({U_BASELINE:g}/{U_PEAK:g})")
    ax_pr.plot(
        t_s, s_lp, color="C0", lw=TRACE_LW,
        label=f"LP(u, {prefilter_ms:g} ms)/max  (τ={tau_t} t)",
    )
    ax_pr.set_ylabel("drive", fontsize=8)
    ax_pr.set_title(
        f"PR drive  onset@ u={_first_change(u)} / s={_first_change(s_lp)}",
        fontsize=9,
    )
    ax_pr.axhline(U_BASELINE, color="0.7", lw=0.5)
    ax_pr.axvline(t_on * dt_ms / 1000.0, color="0.85", lw=0.6, ls="--")
    ax_pr.set_ylim(U_BASELINE - 0.1, U_PEAK + 0.15)
    ax_pr.tick_params(labelsize=7)
    ax_pr.legend(loc="upper right", fontsize=7, frameon=False)

    for row, group in enumerate(groups):
        for col in range(ncols):
            ax = axes[row][col]
            if col >= len(group):
                ax.axis("off")
                continue
            name = str(group[col])
            i = idx_from_cell[name]
            w = with_lp[i]
            wo = without_lp[i]
            ax.plot(
                t_s, w, color="C0", lw=TRACE_LW,
                label=f"with LP {prefilter_ms:g} ms",
            )
            ax.plot(t_s, wo, color="C1", lw=TRACE_LW, label="no drive LP")
            t_w = _first_nz(w)
            t_wo = _first_nz(wo)
            ax.set_title(
                f"{name}  nz@ {t_w}/{t_wo}",
                fontsize=8,
            )
            ax.set_ylim(*TRACE_YLIM)
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.axvline(t_on * dt_ms / 1000.0, color="0.85", lw=0.6, ls="--")
            if row == nrows - 1:
                ax.set_xlabel("t (s)", fontsize=8)
            if col == 0:
                ax.set_ylabel("Ca gt", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"ImpR Ca gt: drive LP {prefilter_ms:g} ms on/off  "
        f"(u={U_BASELINE:g}/{U_PEAK:g}, pulse={ms_pulse:g} ms, t_on={t_on}, Δt={dt_ms:g} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save)
    print(f"saved {save}")
    print(f"  PR u onset={_first_change(u)}  s onset={_first_change(s_lp)}")
    for name in names:
        i = idx_from_cell[name]
        print(
            f"  {name}: first_nz with={_first_nz(with_lp[i])} "
            f"without={_first_nz(without_lp[i])}"
        )
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--ms-pulse", type=float, default=50.0)
    ap.add_argument("--t-on", type=int, default=50)
    ap.add_argument("--n-t", type=int, default=151)
    ap.add_argument("--delta-ms", type=float, default=DELTA_MS)
    ap.add_argument("--prefilter-ms", type=float, default=PREFILTER_MS)
    args = ap.parse_args()

    dt_ms = float(args.delta_ms)
    prefilter_ms = float(args.prefilter_ms)
    u, s_lp = _drive_u_s(
        t_on=args.t_on, n_t=args.n_t, ms_pulse=args.ms_pulse,
        dt_ms=dt_ms, prefilter_ms=prefilter_ms,
    )
    with_lp = _impr_rt(
        t_on=args.t_on, n_t=args.n_t, ms_pulse=args.ms_pulse,
        dt_ms=dt_ms, prefilter_ms=prefilter_ms,
    )
    without_lp = _impr_rt(
        t_on=args.t_on, n_t=args.n_t, ms_pulse=args.ms_pulse,
        dt_ms=dt_ms, prefilter_ms=None,
    )
    _plot(
        with_lp, without_lp, u, s_lp,
        t_on=args.t_on, dt_ms=dt_ms, ms_pulse=float(args.ms_pulse),
        prefilter_ms=prefilter_ms, save=args.save, show=args.show,
    )


if __name__ == "__main__":
    main()
