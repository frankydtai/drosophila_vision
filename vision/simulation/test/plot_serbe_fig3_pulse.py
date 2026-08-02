"""Plot Serbe et al. 2016 Figure 3 filter chain on a 225 ms dark-bar pulse.

Stimulus: baseline 0, value 1 while the bar is on (brightness decrement).
Pipeline: high-pass (tHP) -> half-wave rectification -> low-pass (tLP).

Default time constants are the fitted values from Figure 3 (Tm1–Tm9).

Usage (from ``SimulationCode/``):

    ../.venv/bin/python 6_test/plot_serbe_fig3_pulse.py
    ../.venv/bin/python 6_test/plot_serbe_fig3_pulse.py --cell tm2 --show
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

import blindschleiche_py3 as bs
from plot.utils import save_figure

DEFAULT_SAVE = os.path.join(HERE, "serbe_fig3_pulse.png")
MS_PULSE = 225.0

# Serbe et al. 2016 Neuron Figure 3 fitted time constants (seconds).
SERBE_TM = {
    "tm1": {"t_hp_s": 1.23, "t_lp_s": 0.23},
    "tm2": {"t_hp_s": 0.36, "t_lp_s": 0.10},
    "tm4": {"t_hp_s": 0.25, "t_lp_s": 0.20},
    "tm9": {"t_hp_s": None, "t_lp_s": 0.63},
}


def tau_s_to_samples(tau_s: float | None, dt_ms: float) -> float:
    if tau_s is None:
        return 0.0
    return tau_s * 1000.0 / dt_ms


def dark_bar_pulse(t_ms: np.ndarray, *, t_on_ms: float, ms_pulse: float) -> np.ndarray:
    """1 while the dark bar is on, else 0 (Serbe supplemental stimulus encoding)."""
    t_off_ms = t_on_ms + ms_pulse
    return ((t_ms >= t_on_ms) & (t_ms < t_off_ms)).astype(np.float64)


def serbe_filter_chain(
    stimulus: np.ndarray,
    *,
    t_hp_s: float | None,
    t_lp_s: float,
    dt_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(stimulus, dtype=np.float64)
    tau_hp = tau_s_to_samples(t_hp_s, dt_ms)
    tau_lp = tau_s_to_samples(t_lp_s, dt_ms)

    if tau_hp > 0.0:
        after_hp = bs.highpass(x, tau_hp)
    else:
        after_hp = x.copy()

    after_rec = np.maximum(after_hp, 0.0)
    after_lp = bs.lowpass(after_rec, tau_lp)
    return after_hp, after_rec, after_lp


def lp_only(stimulus: np.ndarray, *, t_lp_s: float, dt_ms: float) -> np.ndarray:
    """Low-pass the raw input, skipping HP and rectification."""
    tau_lp = tau_s_to_samples(t_lp_s, dt_ms)
    return bs.lowpass(np.asarray(stimulus, dtype=np.float64), tau_lp)


def hp_lp_skip_rec(
    stimulus: np.ndarray,
    *,
    t_hp_s: float | None,
    t_lp_s: float,
    dt_ms: float,
) -> np.ndarray:
    """High-pass then low-pass, skipping rectification."""
    after_hp, _, _ = serbe_filter_chain(
        stimulus,
        t_hp_s=t_hp_s,
        t_lp_s=t_lp_s,
        dt_ms=dt_ms,
    )
    tau_lp = tau_s_to_samples(t_lp_s, dt_ms)
    return bs.lowpass(after_hp, tau_lp)


def plot_serbe_pulse(
    path: str,
    *,
    show: bool = False,
    ms_pulse: float = MS_PULSE,
    t_on_ms: float = 500.0,
    t_total_ms: float = 3000.0,
    dt_ms: float = 1.0,
    cells: tuple[str, ...] = ("tm1", "tm2", "tm4", "tm9"),
) -> None:
    n = int(round(t_total_ms / dt_ms)) + 1
    t_ms = np.arange(n, dtype=np.float64) * dt_ms
    stimulus = dark_bar_pulse(t_ms, t_on_ms=t_on_ms, ms_pulse=ms_pulse)

    stage_labels = ("input", "after HP", "after rec", "after LP")
    stage_colors = ("0.15", "C0", "C1", "C2")

    nrows = len(stage_labels)
    ncols = len(cells)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.0 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    legend_done = False

    for col, cell in enumerate(cells):
        params = SERBE_TM[cell]
        after_hp, after_rec, after_lp = serbe_filter_chain(
            stimulus,
            t_hp_s=params["t_hp_s"],
            t_lp_s=params["t_lp_s"],
            dt_ms=dt_ms,
        )
        direct_lp = lp_only(stimulus, t_lp_s=params["t_lp_s"], dt_ms=dt_ms)
        hp_lp = hp_lp_skip_rec(
            stimulus,
            t_hp_s=params["t_hp_s"],
            t_lp_s=params["t_lp_s"],
            dt_ms=dt_ms,
        )
        traces = (stimulus, after_hp, after_rec, after_lp)
        hp_label = "off" if params["t_hp_s"] is None else f"{params['t_hp_s']:.2f} s"
        col_title = f"{cell.upper()}  (tHP={hp_label}, tLP={params['t_lp_s']:.2f} s)"

        for row, (label, y, color) in enumerate(zip(stage_labels, traces, stage_colors)):
            ax = axes[row, col]
            last_row = row == nrows - 1
            ax.plot(
                t_ms / 1000.0,
                y,
                color=color,
                lw=1.6,
                label="HP → rec → LP" if last_row else None,
            )
            ax.axvspan(
                t_on_ms / 1000.0,
                (t_on_ms + ms_pulse) / 1000.0,
                color="0.9",
                zorder=0,
            )
            ax.axhline(0.0, color="0.5", lw=0.6, ls=":")
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.set_title(col_title, fontsize=9)
            if last_row:
                ax.set_xlabel("time (s)")
                ax.plot(
                    t_ms / 1000.0,
                    direct_lp,
                    color=color,
                    lw=1.4,
                    ls="--",
                    label="LP only (skip HP, rec)",
                )
                ax.plot(
                    t_ms / 1000.0,
                    hp_lp,
                    color=color,
                    lw=0.5,
                    ls="none",
                    marker="o",
                    markersize=2,
                    markevery=max(1, int(round(20.0 / dt_ms))),
                    label="HP → LP (skip rec)",
                )
                if not legend_done:
                    ax.legend(loc="upper right", fontsize=6, frameon=False)
                    legend_done = True

    fig.suptitle(
        f"Serbe Fig. 3 filter chain — {ms_pulse:g} ms dark-bar pulse "
        f"(t_on={t_on_ms / 1000:g} s)",
        fontsize=11,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    save_figure(fig, path, dpi=150)
    print(f"saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", default=DEFAULT_SAVE, help="output PNG path")
    p.add_argument("--show", action="store_true")
    p.add_argument("--ms-pulse", type=float, default=MS_PULSE, help="dark-bar duration (ms)")
    p.add_argument("--t-on-ms", type=float, default=500.0, help="pulse onset (ms)")
    p.add_argument("--t-total-ms", type=float, default=3000.0, help="simulation length (ms)")
    p.add_argument("--dt-ms", type=float, default=1.0, help="time step (ms)")
    p.add_argument(
        "--cell",
        choices=sorted(SERBE_TM),
        action="append",
        dest="cells",
        help="plot only selected cell(s); default all four",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.ms_pulse <= 0:
        raise ValueError("--ms-pulse must be > 0")
    if args.dt_ms <= 0:
        raise ValueError("--dt-ms must be > 0")
    cells = tuple(args.cells) if args.cells else tuple(SERBE_TM)
    plot_serbe_pulse(
        args.save,
        show=args.show,
        ms_pulse=args.ms_pulse,
        t_on_ms=args.t_on_ms,
        t_total_ms=args.t_total_ms,
        dt_ms=args.dt_ms,
        cells=cells,
    )


if __name__ == "__main__":
    main()
