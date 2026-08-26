"""Plot simplified sag/rebound model (option 2): constant τ + logistic u_∞(V).

Isolated leak + Ih-like conductance (no network):

    C V̇ = −g_L (V−E_L) − g u (V−E_h) + I
    τ u̇ = u_∞(V) − u

``u_∞``: logistic only. Columns: u_∞ | sweep s | sweep g | sweep τ.

Usage (from SimulationCode/):

    ../.venv/bin/python scratch/plot_sag_curves.py
    ../.venv/bin/python scratch/plot_sag_curves.py --show
    ../.venv/bin/python scratch/plot_sag_curves.py --g-list 0 10 25 50 100 --tau-list 100 250 500 850
    ../.venv/bin/python scratch/plot_sag_curves.py --slope-list -0.05 -0.1 -0.15 -0.2
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import FiveCol_MedSim_Pytorch as fc
from config import NEURON_SCHEMA

P = NEURON_SCHEMA["params"]
from training_config import DELTA_MS

DEFAULT_SAVE = os.path.join(HERE, "sag_curves.png")
DEFAULT_G_LIST = [0.0, 10.0, 25.0, 50.0, 100.0]
DEFAULT_TAU_LIST = [100.0, 250.0, 500.0, 850.0]
DEFAULT_SLOPE_LIST = [-0.05, -0.1, -0.15, -0.2]
DEFAULT_FIXED_G = 50.0


def _init(name: str) -> float:
    return float(P[name]["val"])


def u_inf(Vm, midv, slope):
    """Logistic ON Ih-like gate (slope typically negative → hyperpolarization-activated)."""
    return 1.0 / (1.0 + np.exp((midv - Vm) * slope))


def simulate_pulse(
    gmax,
    *,
    midv,
    slope,
    tau_ms,
    e_leak,
    E_h,
    I_pulse,
    t_total_ms,
    t_pulse_on_ms,
    t_pulse_off_ms,
    settle_ms,
):
    """Option-2 neuron; returns t[ms], Vm, g=gmax*u, u."""
    dt = float(fc.delta_ms)
    n = int(round(t_total_ms / dt))
    n_settle = int(round(settle_ms / dt))
    i_on = int(round(t_pulse_on_ms / dt))
    i_off = int(round(t_pulse_off_ms / dt))
    cdt = float(fc.cdt)
    g_leak = float(fc.g_leak)
    tau = float(tau_ms)

    def step(Vm, u, I):
        u_ss = float(u_inf(Vm, midv, slope))
        u = dt / tau * (u_ss - u) + u
        g = u * gmax
        Vm = (g_leak * e_leak + E_h * g + cdt * Vm + I) / (g + g_leak + cdt)
        return Vm, u, g

    Vm = float(e_leak)
    u = float(u_inf(Vm, midv, slope))
    for _ in range(n_settle):
        Vm, u, _ = step(Vm, u, 0.0)

    t = np.arange(n) * dt
    Vm_trace = np.empty(n)
    g_trace = np.empty(n)
    u_trace = np.empty(n)
    for i in range(n):
        I = I_pulse if i_on <= i < i_off else 0.0
        Vm, u, g = step(Vm, u, I)
        Vm_trace[i] = Vm
        g_trace[i] = g
        u_trace[i] = u
    return t, Vm_trace, g_trace, u_trace


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Vm-lo", type=float, default=-100.0)
    p.add_argument("--Vm-hi", type=float, default=20.0)
    p.add_argument("--n", type=int, default=401)
    p.add_argument("--midv", type=float, default=None, help="half-activation [mV]")
    p.add_argument("--slope", type=float, default=None, help="logistic slope (1/mV) for g/τ sweeps")
    p.add_argument("--tau", type=float, default=250.0, help="constant gate τ [ms] for g- and s-sweep")
    p.add_argument("--tau-list", type=float, nargs="+", default=DEFAULT_TAU_LIST,
                   help="τ values [ms] for τ-sweep panels (fixed g)")
    p.add_argument("--slope-list", type=float, nargs="+", default=DEFAULT_SLOPE_LIST,
                   help="slope values for slope-sweep panels (fixed g,τ)")
    p.add_argument("--fixed-g", type=float, default=DEFAULT_FIXED_G,
                   help="fixed gmax [nS] for τ- and slope-sweep panels")
    p.add_argument("--E-h", type=float, default=None, help="Ih reversal [mV] (default E_Ih)")
    p.add_argument("--g-list", type=float, nargs="+", default=DEFAULT_G_LIST,
                   help="gmax values [nS]")
    p.add_argument("--e-leak", type=float, default=float(fc.E_LEAK_DEPOL))
    p.add_argument("--I-pulse", type=float, default=-40.0)
    p.add_argument("--t-total", type=float, default=2000.0)
    p.add_argument("--t-pulse-on", type=float, default=200.0)
    p.add_argument("--t-pulse-off", type=float, default=1000.0)
    p.add_argument("--settle", type=float, default=5000.0)
    p.add_argument("--save", type=str, default=DEFAULT_SAVE)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    midv = args.midv if args.midv is not None else _init("Ih_midv")
    slope = args.slope if args.slope is not None else _init("Ih_slope")
    E_h = args.E_h if args.E_h is not None else float(fc.E_Ih)
    g_list = list(args.g_list)
    tau_list = list(args.tau_list)
    slope_list = list(args.slope_list)
    g_fix = args.fixed_g

    Vm_axis = np.linspace(args.Vm_lo, args.Vm_hi, args.n)
    fig = plt.figure(figsize=(18.0, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    # col0: u_∞ (span rows); cols: sweep s | sweep g | sweep τ  (V top, gu bottom)
    ax_u = fig.add_subplot(gs[:, 0])
    ax_v_sl = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[0, 2])
    ax_v_tau = fig.add_subplot(gs[0, 3])
    ax_g_sl = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])
    ax_g_tau = fig.add_subplot(gs[1, 3])

    cmap_sl = plt.cm.cividis(np.linspace(0.15, 0.9, len(slope_list)))

    for sl, color in zip(slope_list, cmap_sl):
        ax_u.plot(Vm_axis, u_inf(Vm_axis, midv, sl), color=color, ls="-",
                  label=rf"$s$={sl:g}")
    ax_u.axvline(midv, color="0.5", ls=":", lw=0.8)
    ax_u.set_xlabel(r"$V$ [mV]")
    ax_u.set_ylabel(r"$u_\infty$")
    ax_u.set_ylim(-0.05, 1.05)
    ax_u.set_title(rf"$u_\infty$ (logistic), mid={midv:g} mV")
    ax_u.legend(fontsize=8, loc="best")
    ax_u.grid(True, alpha=0.3)

    common_kwargs = dict(
        midv=midv,
        e_leak=args.e_leak,
        E_h=E_h,
        I_pulse=args.I_pulse,
        t_total_ms=args.t_total,
        t_pulse_on_ms=args.t_pulse_on,
        t_pulse_off_ms=args.t_pulse_off,
        settle_ms=args.settle,
    )

    cmap_g = plt.cm.viridis(np.linspace(0.15, 0.9, len(g_list)))
    for gmax, color in zip(g_list, cmap_g):
        t, Vm, g, u = simulate_pulse(
            gmax, slope=slope, tau_ms=args.tau, **common_kwargs,
        )
        lab = rf"$g$={gmax:g}"
        ax_v.plot(t, Vm, color=color, ls="-", lw=1.5, label=lab)
        ax_g.plot(t, g, color=color, ls="-", lw=1.5, label=lab)

    cmap_tau = plt.cm.plasma(np.linspace(0.15, 0.85, len(tau_list)))
    for tau_ms, color in zip(tau_list, cmap_tau):
        t, Vm, g, u = simulate_pulse(
            g_fix, slope=slope, tau_ms=tau_ms, **common_kwargs,
        )
        lab = rf"$\tau$={tau_ms:g}"
        ax_v_tau.plot(t, Vm, color=color, ls="-", lw=1.5, label=lab)
        ax_g_tau.plot(t, g, color=color, ls="-", lw=1.5, label=lab)

    for sl, color in zip(slope_list, cmap_sl):
        t, Vm, g, u = simulate_pulse(
            g_fix, slope=sl, tau_ms=args.tau, **common_kwargs,
        )
        lab = rf"$s$={sl:g}"
        ax_v_sl.plot(t, Vm, color=color, ls="-", lw=1.5, label=lab)
        ax_g_sl.plot(t, g, color=color, ls="-", lw=1.5, label=lab)

    pulse_axes = (ax_v, ax_g, ax_v_tau, ax_g_tau, ax_v_sl, ax_g_sl)
    for ax in pulse_axes:
        ax.axvspan(args.t_pulse_on, args.t_pulse_off, color="0.85", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax_v.axhline(args.e_leak, color="k", ls=":", lw=0.8)
    ax_v.set_ylabel(r"$V$ [mV]")
    ax_v.set_title(
        rf"sweep $g$, $\tau$={args.tau:g} ms, $s$={slope:g}"
    )
    ax_g.set_xlabel(r"$t$ [ms]")
    ax_g.set_ylabel(r"$g\,u$ [nS]")

    ax_v_tau.axhline(args.e_leak, color="k", ls=":", lw=0.8)
    ax_v_tau.set_ylabel(r"$V$ [mV]")
    ax_v_tau.set_title(
        rf"sweep $\tau$, $g$={g_fix:g} nS, $s$={slope:g}"
    )
    ax_g_tau.set_xlabel(r"$t$ [ms]")
    ax_g_tau.set_ylabel(r"$g\,u$ [nS]")

    ax_v_sl.axhline(args.e_leak, color="k", ls=":", lw=0.8)
    ax_v_sl.set_ylabel(r"$V$ [mV]")
    ax_v_sl.set_title(
        rf"sweep slope, $g$={g_fix:g} nS, $\tau$={args.tau:g} ms"
    )
    ax_g_sl.set_xlabel(r"$t$ [ms]")
    ax_g_sl.set_ylabel(r"$g\,u$ [nS]")

    fig.suptitle(
        rf"$E_h$={E_h:g} mV, $\Delta t$={DELTA_MS:g} ms, $I$={args.I_pulse:g} pA",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save)) or ".", exist_ok=True)
    fig.savefig(args.save, dpi=150)
    print(f"saved {args.save}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
