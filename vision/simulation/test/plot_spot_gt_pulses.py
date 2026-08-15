"""Plot spot RecF gt time courses: 50 ms pulse, 500 ms pulse, step (continue on).

All traces are aligned with ``t_on = 0.5 s`` like ``model_gt_spot.png``.
No photoreceptor (R1-6, R7, R8) panels — gt-cell gt only.

Two PNGs:

1. ``spot_gt_pulses_LTI.png`` — LTI step difference ``p(t) = s(t) - s(t - Δ)``
   on center-bin ImpR×RecF targets from Medulla_Library ``read_RecF_data`` (legacy gt).
2. ``spot_gt_pulses_filter.png`` — drive each cell's IR filter with
   different-w ``u[t]`` (50 ms / 500 ms / continue-on from ``T_ON``).

Usage (from ``SimulationCode/``):

    ../.venv/bin/python test/plot_spot_gt_pulses.py
    ../.venv/bin/python test/plot_spot_gt_pulses.py --show
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

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
import blindschleiche_py3 as bs
from network.build import cell_family_rows, cell_names_in_family_order
from plot.spot import CENTER_BIN, _style_time_axis
from figure.panel import GT_COLOR, TRACE_LW, TRACE_YLIM, save_figure
from const_default import GT_AMP
from training_config import DELTA_MS, IMPULSE_MAXTIME, T_ON, t_from_ms

DEFAULT_SAVE = os.path.join(HERE, "spot_gt_pulses_LTI.png")
DEFAULT_SAVE_FILTER = os.path.join(HERE, "spot_gt_pulses_filter.png")
EXCLUDE_CELLS = frozenset({"R1-6", "R7", "R8"})
PULSE_50_MS = 50.0
PULSE_500_MS = 500.0
LABEL_STEP = "continue on (step)"
LABEL_500 = "500 ms pulse"
LABEL_50 = "50 ms pulse"

# Same IR taus as ``Medulla_Library.read_RecF_ImpR`` (×10 ms).
IR_HP = np.array([39.1, 28.8, 00.0, 38.1, 12.7, 31.8, 26.0, 0.00, 0.00, 29.6, 15.3, 24.9, 0.00])
IR_LP = np.array([03.8, 05.8, 05.4, 02.3, 04.2, 05.4, 02.7, 03.8, 07.7, 04.4, 01.4, 02.4, 10.7])


def pulse_from_step(step: np.ndarray, pulse_t: int) -> np.ndarray:
    """Response to a finite pulse: ``s(t) - s(t - Δ)`` (pre-onset s ≡ 0)."""
    step = np.asarray(step, dtype=np.float64)
    out = np.zeros_like(step)
    d = int(pulse_t)
    for k in range(step.shape[0]):
        k0 = k - d
        out[k] = step[k] - (step[k0] if k0 >= 0 else 0.0)
    return out


def make_u(n_t: int, t_on: int, width_t: int | None) -> np.ndarray:
    """Node pulse/step ``u[t]`` starting at ``t_on``; ``width_t=None`` → continue on."""
    u = np.zeros(n_t, dtype=np.float64)
    if width_t is None:
        u[t_on:] = 1.0
    else:
        u[t_on:t_on + int(width_t)] = 1.0
    return u


def filter_impr_raw(u: np.ndarray, hp: float, lp: float, *, s_max: float, add_l12: bool) -> np.ndarray:
    """IR path matching ``read_RecF_ImpR`` before ``normalize_gt``.

    ``s_max`` is the shared prefilter peak (from the continue-on step) so pulse
    ws share the same gain as ImpR construction.
    """
    s = bs.lowpass(np.asarray(u, dtype=np.float64), 5)
    s = s / float(s_max)
    if hp == 0:
        filtered = bs.lowpass(s, lp)
    else:
        filtered = bs.bandpass(s, hp, lp)
    if add_l12:
        filtered = filtered + 0.4 * s
    return filtered


def absmax_from_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - float(x[0])
    return float(max(abs(np.nanmax(x)), abs(np.nanmin(x))))


def recf_gts() -> dict[str, np.ndarray]:
    """Bright spot gts ``(9, T)`` keyed by gt cell name."""
    raw = ml.read_RecF_data()
    out = {}
    for i, name in enumerate(ml.cell_list):
        out[str(name)] = raw[i] * GT_AMP
    return out


def filter_traces() -> dict[str, dict[str, np.ndarray]]:
    """Center-bin responses from driving IR filters with different-w ``u[t]``.

    Returns ``{cell: {"step", "p500", "p50"}}`` scaled like ``GT_AMP * RecF * ImpR``.
    """
    recf, _ = ml.read_RecF_ImpR()
    n_t = IMPULSE_MAXTIME
    pulse_50 = t_from_ms(PULSE_50_MS)
    pulse_500 = t_from_ms(PULSE_500_MS)
    u_step = make_u(n_t, T_ON, None)
    u_500 = make_u(n_t, T_ON, pulse_500)
    u_50 = make_u(n_t, T_ON, pulse_50)
    s_max = float(np.max(bs.lowpass(u_step, 5)))
    out: dict[str, dict[str, np.ndarray]] = {}
    for i, name in enumerate(ml.cell_list):
        hp = float(IR_HP[i])
        lp = float(IR_LP[i])
        add_l12 = i < 2
        r_step = filter_impr_raw(u_step, hp, lp, s_max=s_max, add_l12=add_l12)
        scale = absmax_from_zero(r_step)
        spatial = float(recf[i, CENTER_BIN * 5 + 2])
        gain = (GT_AMP * spatial / scale) if scale > 0 else 0.0

        def _scaled(u: np.ndarray) -> np.ndarray:
            filtered = filter_impr_raw(u, hp, lp, s_max=s_max, add_l12=add_l12)
            return (filtered - filtered[0]) * gain

        out[str(name)] = {
            "step": _scaled(u_step),
            "p500": _scaled(u_500),
            "p50": _scaled(u_50),
        }
    return out


def _plot_pulse_grid(
    path: str,
    *,
    series_by_name: dict[str, dict[str, np.ndarray]],
    title: str,
    show: bool = False,
) -> None:
    active = [str(n) for n in ml.cell_list if str(n) not in EXCLUDE_CELLS]
    groups = [np.array(row) for row in cell_family_rows(active)]
    names = cell_names_in_family_order(active)
    n_t = IMPULSE_MAXTIME
    t = np.arange(n_t)
    ylo, yhi = TRACE_YLIM

    nrows = len(groups)
    ncols = 5
    fig = plt.figure(figsize=(3.0 * ncols, 2.0 * nrows))
    gs = fig.add_gridspec(
        nrows, ncols,
        hspace=0.55, wspace=0.55, top=0.92, bottom=0.06, left=0.07, right=0.98,
    )
    legend_done = False

    for gi, row_names in enumerate(groups):
        row_idx = [names.index(str(n)) for n in row_names]
        start = (ncols - len(row_idx)) // 2
        for j, ni in enumerate(row_idx):
            name = names[ni]
            series = series_by_name.get(name)
            col = start + j
            ax_time = fig.add_subplot(gs[gi, col])
            if series is None:
                ax_time.axis("off")
                continue

            ax_time.plot(
                t, series["step"], color=GT_COLOR, linewidth=TRACE_LW,
                linestyle="-", label=LABEL_STEP,
            )
            ax_time.plot(
                t, series["p500"], color=GT_COLOR, linewidth=TRACE_LW,
                linestyle="--", label=LABEL_500,
            )
            ax_time.plot(
                t, series["p50"], color=GT_COLOR, linewidth=0.5,
                linestyle="none", marker="o", markersize=1, markevery=3,
                label=LABEL_50,
            )
            ax_time.set_title(name, fontsize=8, pad=2)
            ax_time.set_ylim(ylo, yhi)
            _style_time_axis(ax_time, show_xlabel=True, n_t=n_t)
            if j == 0:
                ax_time.set_ylabel("mV", fontsize=7)
            ax_time.tick_params(labelsize=6)
            ax_time.axhline(0.0, color="0.4", linewidth=0.6, linestyle=":", zorder=0)
            if not legend_done:
                ax_time.legend(loc="upper right", fontsize=6, frameon=False)
                legend_done = True

    fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_figure(fig, path, dpi=150)
    print(f"saved {path}")
    if show:
        img = plt.imread(path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def plot_gt_pulses(path: str, *, show: bool = False) -> None:
    """PNG 1: LTI pulse-from-step on Medulla_Library RecF gt center traces."""
    rts = recf_gts()
    pulse_50 = t_from_ms(PULSE_50_MS)
    pulse_500 = t_from_ms(PULSE_500_MS)
    series = {}
    for name, rt in rts.items():
        step = np.asarray(rt[CENTER_BIN], dtype=np.float64)
        series[name] = {
            "step": step,
            "p500": pulse_from_step(step, pulse_500),
            "p50": pulse_from_step(step, pulse_50),
        }
    t_on_s = fc.t_on * DELTA_MS / 1000.0
    _plot_pulse_grid(
        path,
        series_by_name=series,
        title=(
            f"spot bright gt  (t_on={t_on_s:g} s; "
            f"step | {int(PULSE_500_MS)} ms pulse | {int(PULSE_50_MS)} ms pulse; "
            f"LTI s(t)-s(t-Δ))"
        ),
        show=show,
    )


def plot_gt_pulses_filter(path: str, *, show: bool = False) -> None:
    """PNG 2: IR filter responses to different-w ``u[t]`` from ``T_ON``."""
    series = filter_traces()
    t_on_s = fc.t_on * DELTA_MS / 1000.0
    _plot_pulse_grid(
        path,
        series_by_name=series,
        title=(
            f"spot bright gt  (t_on={t_on_s:g} s; "
            f"step | {int(PULSE_500_MS)} ms pulse | {int(PULSE_50_MS)} ms pulse; "
            f"filter u[t])"
        ),
        show=show,
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", default=DEFAULT_SAVE, help="LTI-difference PNG path")
    p.add_argument(
        "--save-filter", default=DEFAULT_SAVE_FILTER,
        help="filter-driven u[t] PNG path",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_gt_pulses(args.save, show=False)
    plot_gt_pulses_filter(args.save_filter, show=args.show)


if __name__ == "__main__":
    main()
