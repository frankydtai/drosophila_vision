"""Plot conductance Ih curves and pulse responses vs gmax / constant tau (no network).

Formulas match ``FiveCol_MedSim_Pytorch._ih_gate_step`` / ``update_Vm``.

Usage (from SimulationCode/):

    ../.venv/bin/python scratch/plot_h_curves.py
    ../.venv/bin/python scratch/plot_h_curves.py --show
    ../.venv/bin/python scratch/plot_h_curves.py --gmax-list 0 10 25 50 100
    ../.venv/bin/python scratch/plot_h_curves.py --tau-const-list 100 250 500 850
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

DEFAULT_SAVE = os.path.join(HERE, "h_curves.png")
DEFAULT_PULSE_SAVE = os.path.join(HERE, "h_pulse_gmax.png")
DEFAULT_PULSE_TAU_SAVE = os.path.join(HERE, "h_pulse_tau_const.png")
DEFAULT_GMAX_LIST = [0.0, 10.0, 25.0, 50.0, 100.0]
# Floor / intermediate / peak of voltage-dependent τ(V) (ms).
DEFAULT_TAU_CONST_LIST = [100.0, 250.0, 500.0, 850.0]
TAU_VDEP_MAX_MS = 850.0  # at V = tau_mid


def _init(name: str) -> float:
    return float(P[name]["val"])


def ih_ss(Vm, midv, slope):
    """Steady gate; OFF channel passes ``slope = -Ih_slope_off``."""
    return 1.0 / (1.0 + np.exp((midv - Vm) * slope))


def ih_tau_ms(Vm, tau_midv):
    """Voltage-dependent gate time constant [ms]."""
    return (
        1.5
        / (np.exp(-0.1 * (Vm - tau_midv)) + np.exp(+0.1 * (Vm - tau_midv)))
        * 1000.0
        + 100.0
    )


def _tau_of(Vm, tau_mid, tau_const):
    """Voltage-dependent τ unless ``tau_const`` is set (ms)."""
    if tau_const is not None:
        return float(tau_const)
    return float(ih_tau_ms(Vm, tau_mid))


def simulate_pulse(
    gmax,
    *,
    midv,
    slope,
    tau_mid,
    e_leak,
    I_pulse,
    t_total_ms,
    t_pulse_on_ms,
    t_pulse_off_ms,
    settle_ms,
    tau_const=None,
):
    """Isolated leak+ON-Ih neuron; returns t[ms], Vm, g_Ih.

    ``tau_const``: if set, gate uses fixed τ [ms]; else voltage-dependent τ(V).
    """
    dt = float(fc.delta_ms)
    n = int(round(t_total_ms / dt))
    n_settle = int(round(settle_ms / dt))
    i_on = int(round(t_pulse_on_ms / dt))
    i_off = int(round(t_pulse_off_ms / dt))
    cdt = float(fc.cdt)
    g_leak = float(fc.g_leak)
    E_Ih = float(fc.E_Ih)
    gain = float(fc.Ih_gain)

    Vm = float(e_leak)
    u = float(ih_ss(Vm, midv, slope))
    # settle at I=0
    for _ in range(n_settle):
        u_ss = ih_ss(Vm, midv, slope)
        tau = _tau_of(Vm, tau_mid, tau_const)
        u = dt / tau * (u_ss - u) + u
        g = u * gmax * gain
        Vm = (g_leak * e_leak + E_Ih * g + cdt * Vm) / (g + g_leak + cdt)

    t = np.arange(n) * dt
    Vm_trace = np.empty(n)
    g_trace = np.empty(n)
    for i in range(n):
        I = I_pulse if i_on <= i < i_off else 0.0
        u_ss = ih_ss(Vm, midv, slope)
        tau = _tau_of(Vm, tau_mid, tau_const)
        u = dt / tau * (u_ss - u) + u
        g = u * gmax * gain
        Vm = (g_leak * e_leak + E_Ih * g + cdt * Vm + I) / (g + g_leak + cdt)
        Vm_trace[i] = Vm
        g_trace[i] = g
    return t, Vm_trace, g_trace


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Vm-lo", type=float, default=-100.0, help="Vm axis low [mV]")
    p.add_argument("--Vm-hi", type=float, default=20.0, help="Vm axis high [mV]")
    p.add_argument("--n", type=int, default=401, help="number of Vm samples")
    p.add_argument("--Ih-midv", type=float, default=None)
    p.add_argument("--Ih-slope", type=float, default=None)
    p.add_argument("--tau-midv", type=float, default=None)
    p.add_argument("--Ih-midv-off", type=float, default=None)
    p.add_argument("--Ih-slope-off", type=float, default=None)
    p.add_argument("--tau-midv-off", type=float, default=None)
    p.add_argument("--Ih-gmax", type=float, default=None, help="gmax for I–V panel [nS]")
    p.add_argument("--Ih-gmax-off", type=float, default=None)
    p.add_argument(
        "--gmax-list",
        type=float,
        nargs="+",
        default=DEFAULT_GMAX_LIST,
        help="ON gmax values [nS] for pulse panel",
    )
    p.add_argument(
        "--tau-const-list",
        type=float,
        nargs="+",
        default=DEFAULT_TAU_CONST_LIST,
        help="constant τ values [ms] for tau-const pulse panel",
    )
    p.add_argument(
        "--pulse-gmax",
        type=float,
        default=None,
        help="fixed gmax [nS] for tau-const pulse panel (default: Ih_gmax init)",
    )
    p.add_argument("--e-leak", type=float, default=float(fc.E_LEAK_DEPOL),
                   help="leak reversal [mV] (default L1–L3 depolarized)")
    p.add_argument("--I-pulse", type=float, default=-40.0,
                   help="injected current during pulse [pA] (negative=hyperpolarizing)")
    p.add_argument("--t-total", type=float, default=2000.0, help="pulse sim length [ms]")
    p.add_argument("--t-pulse-on", type=float, default=200.0, help="pulse start [ms]")
    p.add_argument("--t-pulse-off", type=float, default=1000.0, help="pulse end [ms]")
    p.add_argument("--settle", type=float, default=5000.0, help="pre-pulse settle [ms]")
    p.add_argument("--save", type=str, default=DEFAULT_SAVE)
    p.add_argument("--save-pulse", type=str, default=DEFAULT_PULSE_SAVE)
    p.add_argument("--save-pulse-tau", type=str, default=DEFAULT_PULSE_TAU_SAVE)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def plot_static_curves(args, midv, slope, tau_mid, midv_off, slope_off, tau_mid_off, gmax, gmax_off):
    Vm = np.linspace(args.Vm_lo, args.Vm_hi, args.n)
    u_on = ih_ss(Vm, midv, slope)
    u_off = ih_ss(Vm, midv_off, -slope_off)
    tau_on = ih_tau_ms(Vm, tau_mid)
    tau_off = ih_tau_ms(Vm, tau_mid_off)
    g_on = gmax * float(fc.Ih_gain)
    g_off = gmax_off * float(fc.Ih_gain)
    I_on = g_on * (float(fc.E_Ih) - Vm)
    I_off = g_off * (float(fc.E_IH_OFF) - Vm)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(Vm, u_on, label=rf"ON  mid={midv:g}, s={slope:g}", color="C0")
    ax.plot(Vm, u_off, label=rf"OFF mid={midv_off:g}, s_eff={-slope_off:g}", color="C1")
    ax.axvline(midv, color="C0", ls=":", lw=0.8)
    ax.axvline(midv_off, color="C1", ls=":", lw=0.8)
    ax.set_xlabel(r"$V_m$ [mV]")
    ax.set_ylabel(r"$u_\infty$")
    ax.set_title("steady gate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(Vm, tau_on, label=rf"ON  $\tau_{{\mathrm{{mid}}}}$={tau_mid:g}", color="C0")
    ax.plot(Vm, tau_off, label=rf"OFF $\tau_{{\mathrm{{mid}}}}$={tau_mid_off:g}", color="C1")
    ax.axvline(tau_mid, color="C0", ls=":", lw=0.8)
    ax.axvline(tau_mid_off, color="C1", ls=":", lw=0.8)
    ax.set_xlabel(r"$V_m$ [mV]")
    ax.set_ylabel(r"$\tau$ [ms]")
    ax.set_title("gate time constant")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(Vm, I_on, label=rf"ON  $E$={fc.E_Ih:g}, $g_{{\max}}$={gmax:g}", color="C0")
    ax.plot(Vm, I_off, label=rf"OFF $E$={fc.E_IH_OFF:g}, $g_{{\max}}$={gmax_off:g}", color="C1")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(float(fc.E_Ih), color="C0", ls=":", lw=0.8)
    ax.axvline(float(fc.E_IH_OFF), color="C1", ls=":", lw=0.8)
    ax.set_xlabel(r"$V_m$ [mV]")
    ax.set_ylabel(r"$I$ at $u=1$ [pA]")
    ax.set_title(r"$I=g(E-V)$ (open gate)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.suptitle("conductance Ih (config init)", fontsize=11)
    return fig


def plot_pulse_gmax(args, midv, slope, tau_mid, gmax_list):
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True, constrained_layout=True)
    ax_v, ax_g = axes
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(gmax_list)))

    for gmax, color in zip(gmax_list, cmap):
        t, Vm, g = simulate_pulse(
            gmax,
            midv=midv,
            slope=slope,
            tau_mid=tau_mid,
            e_leak=args.e_leak,
            I_pulse=args.I_pulse,
            t_total_ms=args.t_total,
            t_pulse_on_ms=args.t_pulse_on,
            t_pulse_off_ms=args.t_pulse_off,
            settle_ms=args.settle,
        )
        ax_v.plot(t, Vm, color=color, label=rf"$g_{{\max}}$={gmax:g}")
        ax_g.plot(t, g, color=color, label=rf"$g_{{\max}}$={gmax:g}")

    for ax in axes:
        ax.axvspan(args.t_pulse_on, args.t_pulse_off, color="0.85", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax_v.axhline(args.e_leak, color="k", ls=":", lw=0.8)
    ax_v.set_ylabel(r"$V_m$ [mV]")
    ax_v.set_title(
        rf"hyperpolarizing pulse $I$={args.I_pulse:g} pA, "
        rf"$E_{{\mathrm{{leak}}}}$={args.e_leak:g} mV, $\Delta t$={DELTA_MS:g} ms"
    )
    ax_g.set_xlabel(r"$t$ [ms]")
    ax_g.set_ylabel(r"$g_h$ [nS]")
    ax_g.set_title(r"Ih conductance during pulse")
    fig.suptitle(r"pulse response vs $g_{\max}$ (ON Ih only)", fontsize=11)
    return fig


def _pulse_kwargs(args, midv, slope, tau_mid, gmax, tau_const=None):
    return dict(
        gmax=gmax,
        midv=midv,
        slope=slope,
        tau_mid=tau_mid,
        e_leak=args.e_leak,
        I_pulse=args.I_pulse,
        t_total_ms=args.t_total,
        t_pulse_on_ms=args.t_pulse_on,
        t_pulse_off_ms=args.t_pulse_off,
        settle_ms=args.settle,
        tau_const=tau_const,
    )


def plot_pulse_tau_const(args, midv, slope, tau_mid, gmax, tau_const_list):
    """Voltage-dependent τ(V) vs fixed τ (same gmax / pulse)."""
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True, constrained_layout=True)
    ax_v, ax_g = axes

    t, Vm, g = simulate_pulse(**_pulse_kwargs(args, midv, slope, tau_mid, gmax))
    ax_v.plot(t, Vm, color="k", lw=2.0, label=r"$\tau(V)$ (voltage-dep.)")
    ax_g.plot(t, g, color="k", lw=2.0, label=r"$\tau(V)$ (voltage-dep.)")

    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(tau_const_list)))
    for tau_c, color in zip(tau_const_list, cmap):
        t, Vm, g = simulate_pulse(
            **_pulse_kwargs(args, midv, slope, tau_mid, gmax, tau_const=tau_c),
        )
        ax_v.plot(t, Vm, color=color, label=rf"$\tau$={tau_c:g} ms (const)")
        ax_g.plot(t, g, color=color, label=rf"$\tau$={tau_c:g} ms (const)")

    for ax in axes:
        ax.axvspan(args.t_pulse_on, args.t_pulse_off, color="0.85", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax_v.axhline(args.e_leak, color="k", ls=":", lw=0.8)
    ax_v.set_ylabel(r"$V_m$ [mV]")
    ax_v.set_title(
        rf"$g_{{\max}}$={gmax:g} nS, $I$={args.I_pulse:g} pA, "
        rf"$E_{{\mathrm{{leak}}}}$={args.e_leak:g} mV "
        rf"(peak $\tau(V)$={TAU_VDEP_MAX_MS:g} ms)"
    )
    ax_g.set_xlabel(r"$t$ [ms]")
    ax_g.set_ylabel(r"$g_h$ [nS]")
    ax_g.set_title(r"Ih conductance during pulse")
    fig.suptitle(r"pulse response: $\tau(V)$ vs constant $\tau$", fontsize=11)
    return fig


def _savefig(fig, path, show):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    midv = args.Ih_midv if args.Ih_midv is not None else _init("Ih_midv")
    slope = args.Ih_slope if args.Ih_slope is not None else _init("Ih_slope")
    tau_mid = args.tau_midv if args.tau_midv is not None else _init("tau_midv")
    midv_off = args.Ih_midv_off if args.Ih_midv_off is not None else _init("Ih_midv_off")
    slope_off = args.Ih_slope_off if args.Ih_slope_off is not None else _init("Ih_slope_off")
    tau_mid_off = args.tau_midv_off if args.tau_midv_off is not None else _init("tau_midv_off")
    gmax = args.Ih_gmax if args.Ih_gmax is not None else _init("Ih_gmax")
    gmax_off = args.Ih_gmax_off if args.Ih_gmax_off is not None else _init("Ih_gmax_off")
    gmax_list = list(args.gmax_list)
    tau_const_list = list(args.tau_const_list)
    pulse_gmax = args.pulse_gmax if args.pulse_gmax is not None else gmax

    fig = plot_static_curves(
        args, midv, slope, tau_mid, midv_off, slope_off, tau_mid_off, gmax, gmax_off,
    )
    _savefig(fig, args.save, args.show)

    fig_p = plot_pulse_gmax(args, midv, slope, tau_mid, gmax_list)
    _savefig(fig_p, args.save_pulse, args.show)

    fig_t = plot_pulse_tau_const(args, midv, slope, tau_mid, pulse_gmax, tau_const_list)
    _savefig(fig_t, args.save_pulse_tau, args.show)


if __name__ == "__main__":
    main()
