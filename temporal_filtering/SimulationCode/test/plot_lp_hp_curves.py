"""Plot passive membrane LP-only vs HP-then-LP on a pulse input.

LP-only (membrane low-pass):

    τ_lp dV/dt = -(V - V_rest) + G S_in(t)

HP + LP (slow average a, high-pass drive into membrane):

    τ_HP da/dt = S_in - a
    S_HP = S_in - a
    τ_lp dV/dt = -(V - V_rest) + G S_HP

Rows: input | HP stage | baseline LP vs HP→LP | sweep G | sweep τ_HP | sweep pulse width.
Columns: left = +pulse (S0), right = −pulse (−S0).

Uses ``blindschleiche_py3.lowpass`` / ``highpass`` (tau in samples).

Usage (from ``SimulationCode/``):

    ../.venv/bin/python 6_test/plot_lp_hp_curves.py
    ../.venv/bin/python 6_test/plot_lp_hp_curves.py --show
    ../.venv/bin/python 6_test/plot_lp_hp_curves.py --gain-list 0.5,1,2 --tau-hp-list 80,200,500,2000 --pulse-list 50,100,500
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

import network_bootstrap  # noqa: F401 — connectome_io on sys.path
import blindschleiche_py3 as bs
from connectome_io import parse_comma_list
from plot.utils import save_figure

DEFAULT_SAVE = os.path.join(HERE, "lp_hp_curves.png")
DEFAULT_GAIN_LIST = "0.5,1,2"
DEFAULT_TAU_HP_LIST = "80,200,500,2000"
DEFAULT_PULSE_LIST = "50,100,500"


def tau_ms_to_samples(tau_ms: float, dt_ms: float) -> float:
    return float(tau_ms) / float(dt_ms)


def make_pulse(t_ms: np.ndarray, *, t_on_ms: float, pulse_ms: float, s0: float) -> np.ndarray:
    t_off_ms = t_on_ms + pulse_ms
    return np.where((t_ms >= t_on_ms) & (t_ms < t_off_ms), s0, 0.0).astype(np.float64)


def membrane_lp(s_in: np.ndarray, *, tau_lp_samples: float, v_rest: float, gain: float) -> np.ndarray:
    """V = V_rest + G * lowpass(S_in)."""
    return v_rest + gain * bs.lowpass(np.asarray(s_in, dtype=np.float64), tau_lp_samples)


def membrane_hp_lp(
    s_in: np.ndarray,
    *,
    tau_hp_samples: float,
    tau_lp_samples: float,
    v_rest: float,
    gain: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (V, a, S_HP) for HP-then-membrane-LP."""
    x = np.asarray(s_in, dtype=np.float64)
    if tau_hp_samples < 1.0:
        a = np.zeros_like(x)
        s_hp = x.copy()
    else:
        a = bs.lowpass(x, tau_hp_samples)
        s_hp = x - a
    v = v_rest + gain * bs.lowpass(s_hp, tau_lp_samples)
    return v, a, s_hp


def _shade_and_grid(ax, t_on_s: float, t_off_s: float, v_rest: float) -> None:
    ax.axvspan(t_on_s, t_off_s, color="0.92", zorder=0)
    ax.axhline(v_rest, color="0.5", lw=0.6, ls=":")
    ax.grid(True, alpha=0.25)


def _cmap_colors(n: int):
    cmap = plt.cm.viridis
    if n <= 1:
        return [cmap(0.55)]
    return [cmap(0.15 + 0.75 * (i / (n - 1))) for i in range(n)]


def _fill_column(
    axes_col,
    *,
    t_ms: np.ndarray,
    t_s: np.ndarray,
    t_on_ms: float,
    pulse_ms: float,
    s0: float,
    v_rest: float,
    gain: float,
    tau_lp: float,
    tau_hp: float,
    tau_lp_ms: float,
    tau_hp_ms: float,
    dt_ms: float,
    gain_list: list[float],
    tau_hp_list: list[float],
    pulse_list: list[float],
    show_ylabel: bool,
    show_legend: bool,
    col_title: str,
) -> None:
    ax0, ax1, ax2, ax3, ax4, ax5 = axes_col
    t_on_s = t_on_ms / 1000.0
    t_off_s = (t_on_ms + pulse_ms) / 1000.0
    s_in = make_pulse(t_ms, t_on_ms=t_on_ms, pulse_ms=pulse_ms, s0=s0)

    v_lp = membrane_lp(s_in, tau_lp_samples=tau_lp, v_rest=v_rest, gain=gain)
    v_hp_lp, a, s_hp = membrane_hp_lp(
        s_in,
        tau_hp_samples=tau_hp,
        tau_lp_samples=tau_lp,
        v_rest=v_rest,
        gain=gain,
    )

    for ax in (ax0, ax1, ax2, ax3, ax4):
        _shade_and_grid(ax, t_on_s, t_off_s, v_rest if ax in (ax2, ax3, ax4) else 0.0)
    ax5.axhline(v_rest, color="0.5", lw=0.6, ls=":")
    ax5.axvline(t_on_s, color="0.6", lw=0.7, ls="--")
    ax5.grid(True, alpha=0.25)

    ax0.plot(t_s, s_in, color="0.15", lw=1.6, label=r"$S_{\mathrm{in}}$")
    ax0.set_title(col_title, fontsize=10)
    if show_legend:
        ax0.legend(loc="upper right", fontsize=7, frameon=False)

    ax1.plot(t_s, a, color="C0", lw=1.4, label=r"$a$")
    ax1.plot(t_s, s_hp, color="C1", lw=1.4, label=r"$S_{\mathrm{HP}}$")
    if show_legend:
        ax1.legend(loc="upper right", fontsize=7, frameon=False)

    ax2.plot(t_s, v_lp, color="C2", lw=1.8, label="LP only")
    ax2.plot(t_s, v_hp_lp, color="C3", lw=1.8, label="HP → LP")
    if show_legend:
        ax2.legend(loc="upper right", fontsize=7, frameon=False)

    for g, color in zip(gain_list, _cmap_colors(len(gain_list))):
        v, _, _ = membrane_hp_lp(
            s_in,
            tau_hp_samples=tau_hp,
            tau_lp_samples=tau_lp,
            v_rest=v_rest,
            gain=g,
        )
        ax3.plot(t_s, v, color=color, lw=1.6, label=rf"$G$={g:g}")
    ax3.set_title(
        rf"sweep $G$ ($\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms)",
        fontsize=8,
    )
    if show_legend:
        ax3.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(gain_list)))

    for thp, color in zip(tau_hp_list, _cmap_colors(len(tau_hp_list))):
        v, _, _ = membrane_hp_lp(
            s_in,
            tau_hp_samples=tau_ms_to_samples(thp, dt_ms),
            tau_lp_samples=tau_lp,
            v_rest=v_rest,
            gain=gain,
        )
        ax4.plot(t_s, v, color=color, lw=1.6, label=rf"$\tau_{{\mathrm{{HP}}}}$={thp:g}")
    ax4.set_title(rf"sweep $\tau_{{\mathrm{{HP}}}}$ ($G$={gain:g})", fontsize=8)
    if show_legend:
        ax4.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(tau_hp_list)))

    for pw, color in zip(pulse_list, _cmap_colors(len(pulse_list))):
        s_pw = make_pulse(t_ms, t_on_ms=t_on_ms, pulse_ms=pw, s0=s0)
        v, _, _ = membrane_hp_lp(
            s_pw,
            tau_hp_samples=tau_hp,
            tau_lp_samples=tau_lp,
            v_rest=v_rest,
            gain=gain,
        )
        ax5.axvspan(t_on_s, (t_on_ms + pw) / 1000.0, color=color, alpha=0.08, zorder=0)
        ax5.plot(t_s, v, color=color, lw=1.6, label=rf"$T$={pw:g} ms")
    ax5.set_title(
        rf"sweep pulse ($G$={gain:g}, $\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms)",
        fontsize=8,
    )
    ax5.set_xlabel("time (s)")
    if show_legend:
        ax5.legend(loc="upper right", fontsize=6, frameon=False, ncol=min(3, len(pulse_list)))

    if show_ylabel:
        ax0.set_ylabel("input")
        ax1.set_ylabel("HP stage")
        ax2.set_ylabel(r"$V$")
        ax3.set_ylabel(r"$V$ (sweep $G$)")
        ax4.set_ylabel(r"$V$ (sweep $\tau_{\mathrm{HP}}$)")
        ax5.set_ylabel(r"$V$ (sweep pulse)")


def plot_lp_hp(
    path: str,
    *,
    show: bool = False,
    dt_ms: float = 1.0,
    t_total_ms: float = 3000.0,
    t_on_ms: float = 500.0,
    pulse_ms: float = 1500.0,
    s0: float = 1.0,
    v_rest: float = 0.0,
    gain: float = 1.0,
    tau_lp_ms: float = 50.0,
    tau_hp_ms: float = 200.0,
    gain_list: list[float] | None = None,
    tau_hp_list: list[float] | None = None,
    pulse_list: list[float] | None = None,
) -> None:
    if gain_list is None:
        gain_list = [float(x) for x in parse_comma_list(DEFAULT_GAIN_LIST)]
    if tau_hp_list is None:
        tau_hp_list = [float(x) for x in parse_comma_list(DEFAULT_TAU_HP_LIST)]
    if pulse_list is None:
        pulse_list = [float(x) for x in parse_comma_list(DEFAULT_PULSE_LIST)]

    n = int(round(t_total_ms / dt_ms)) + 1
    t_ms = np.arange(n, dtype=np.float64) * dt_ms
    t_s = t_ms / 1000.0
    tau_lp = tau_ms_to_samples(tau_lp_ms, dt_ms)
    tau_hp = tau_ms_to_samples(tau_hp_ms, dt_ms)

    fig, axes = plt.subplots(
        6,
        2,
        figsize=(12.0, 12.5),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )

    common = dict(
        t_ms=t_ms,
        t_s=t_s,
        t_on_ms=t_on_ms,
        pulse_ms=pulse_ms,
        v_rest=v_rest,
        gain=gain,
        tau_lp=tau_lp,
        tau_hp=tau_hp,
        tau_lp_ms=tau_lp_ms,
        tau_hp_ms=tau_hp_ms,
        dt_ms=dt_ms,
        gain_list=gain_list,
        tau_hp_list=tau_hp_list,
        pulse_list=pulse_list,
    )
    param_line = (
        rf"pulse={pulse_ms:g} ms, $\tau_lp$={tau_lp_ms:g} ms, "
        rf"$\tau_{{\mathrm{{HP}}}}$={tau_hp_ms:g} ms, $G$={gain:g}, "
        rf"$V_{{\mathrm{{rest}}}}$={v_rest:g}, $|S_0|$={abs(s0):g}"
    )
    _fill_column(
        axes[:, 0],
        s0=abs(s0),
        show_ylabel=True,
        show_legend=True,
        col_title=rf"+pulse ($S_0$={abs(s0):g})",
        **common,
    )
    _fill_column(
        axes[:, 1],
        s0=-abs(s0),
        show_ylabel=False,
        show_legend=False,
        col_title=rf"−pulse ($S_0$={-abs(s0):g})",
        **common,
    )
    axes[0, 0].annotate(
        param_line,
        xy=(0.0, 1.18),
        xycoords="axes fraction",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    fig.suptitle("Passive membrane: LP-only vs HP-then-LP (+pulse | −pulse)", fontsize=12)
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
    p.add_argument("--dt-ms", type=float, default=1.0)
    p.add_argument("--t-total-ms", type=float, default=3000.0)
    p.add_argument("--t-on-ms", type=float, default=500.0)
    p.add_argument("--pulse-ms", type=float, default=1500.0)
    p.add_argument("--s0", type=float, default=1.0, help="pulse amplitude (|S0|; right column uses −|S0|)")
    p.add_argument("--v-rest", type=float, default=0.0)
    p.add_argument("--gain", type=float, default=1.0, help="baseline G (also used in τ_HP sweep)")
    p.add_argument("--tau-lp", type=float, default=50.0, help="LP membrane tau [ms]")
    p.add_argument("--tau-hp", type=float, default=200.0, help="baseline high-pass tau [ms] (also used in G sweep)")
    p.add_argument("--gain-list", type=str, default=DEFAULT_GAIN_LIST, help="comma-separated G sweep")
    p.add_argument("--tau-hp-list", type=str, default=DEFAULT_TAU_HP_LIST, help="comma-separated τ_HP sweep [ms]")
    p.add_argument("--pulse-list", type=str, default=DEFAULT_PULSE_LIST, help="comma-separated pulse widths [ms]")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    for name, val in (
        ("--dt-ms", args.dt_ms),
        ("--t-total-ms", args.t_total_ms),
        ("--pulse-ms", args.pulse_ms),
        ("--tau-lp", args.tau_lp),
        ("--tau-hp", args.tau_hp),
    ):
        if val <= 0:
            raise ValueError(f"{name} must be > 0")
    gain_list = [float(x) for x in parse_comma_list(args.gain_list)]
    tau_hp_list = [float(x) for x in parse_comma_list(args.tau_hp_list)]
    pulse_list = [float(x) for x in parse_comma_list(args.pulse_list)]
    if not gain_list:
        raise ValueError("--gain-list must be non-empty")
    if not tau_hp_list:
        raise ValueError("--tau-hp-list must be non-empty")
    if not pulse_list:
        raise ValueError("--pulse-list must be non-empty")
    if any(g < 0 for g in gain_list):
        raise ValueError("--gain-list values must be >= 0")
    if any(t <= 0 for t in tau_hp_list):
        raise ValueError("--tau-hp-list values must be > 0")
    if any(t <= 0 for t in pulse_list):
        raise ValueError("--pulse-list values must be > 0")
    plot_lp_hp(
        args.save,
        show=args.show,
        dt_ms=args.dt_ms,
        t_total_ms=args.t_total_ms,
        t_on_ms=args.t_on_ms,
        pulse_ms=args.pulse_ms,
        s0=args.s0,
        v_rest=args.v_rest,
        gain=args.gain,
        tau_lp_ms=args.tau_lp,
        tau_hp_ms=args.tau_hp,
        gain_list=gain_list,
        tau_hp_list=tau_hp_list,
        pulse_list=pulse_list,
    )


if __name__ == "__main__":
    main()
