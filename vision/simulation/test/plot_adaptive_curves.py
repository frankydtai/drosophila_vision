"""Plot isolated adaptive temporal-filter neuron (no network).

Formulas match ``FiveCol_MedSim_Pytorch.update_state_adaptive``:

    X = bias + x_t
    g = (X_gate - pivot) * adapt_gain
    τ_r d r/dt = -r + X
    τ_m d v_s/dt = -v_s + (1 - g ρ) X
    τ_m d v_t/dt = -v_t - g (1-ρ) s,   s = r or 1-r from sign(g)
    a = v_s + v_t

Layout (like ``plot_sag_curves.py``): contrast gate | sweep adapt_gain | sweep tau_adapt.
Top row: activity; bottom row: v_transient.

Usage (from SimulationCode/):

    ../.venv/bin/python 6_test/plot_adaptive_curves.py
    ../.venv/bin/python 6_test/plot_adaptive_curves.py --show
    ../.venv/bin/python 6_test/plot_adaptive_curves.py --gadapt-list 0,0.5,1,2 --tau-adapt-list 100,250,500,850
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
from import_bootstrap import parse_comma_list
from const_default import P
from training_config import DELTA_MS

DEFAULT_SAVE = os.path.join(HERE, "adaptive_curves.png")
DEFAULT_GADAPT_LIST = "0,0.5,1,2"
DEFAULT_TAU_ADAPT_LIST = "100,250,500,850"
DEFAULT_FIXED_GADAPT = 1.0
DEFAULT_FIXED_TAU_ADAPT = 100.0


def _init(name: str) -> float:
    return float(P[name]["val"])


def contrast_gate(X_gate, adapt_gain, gate_pivot):
    """Contrast gate ``g = (X_gate - pivot) * adapt_gain``."""
    return (X_gate - gate_pivot) * adapt_gain


def update_step(v_s, v_t, drive_lp, x_t, x_delayed, *, bias, adapt_gain, tau_m, tau_adapt, gate_pivot, dt):
    """One explicit-Euler step; returns activity, v_s, v_t, drive_lp, gate."""
    tau = max(float(tau_m), dt)
    tau_r = max(float(tau_adapt), dt)
    ratio = tau / tau_r
    X = bias + x_t
    X_gate = bias + x_delayed
    gate = (X_gate - gate_pivot) * adapt_gain
    gate_src = drive_lp if gate >= 0.0 else 1.0 - drive_lp

    drive_lp = drive_lp + dt / tau_r * (-drive_lp + X)
    v_s = v_s + dt / tau * (-v_s + (1.0 - gate * ratio) * X)
    v_t = v_t + dt / tau * (-v_t + (-gate * (1.0 - ratio) * gate_src))
    activity = v_s + v_t
    return activity, v_s, v_t, drive_lp, gate


def simulate_pulse(
    adapt_gain,
    *,
    tau_adapt,
    tau_m,
    bias,
    gate_pivot,
    x_base,
    x_pulse,
    t_total_ms,
    t_pulse_on_ms,
    t_pulse_off_ms,
    settle_ms,
    gate_lag=None,
):
    """Isolated adaptive node; returns t[ms], activity, v_s, v_t, drive_lp, gate."""
    dt = float(fc.delta_ms)
    gate_lag = int(fc.gate_lag if gate_lag is None else gate_lag)
    n = int(round(t_total_ms / dt))
    n_settle = int(round(settle_ms / dt))
    i_on = int(round(t_pulse_on_ms / dt))
    i_off = int(round(t_pulse_off_ms / dt))

    def x_at(step_i):
        return x_pulse if i_on <= step_i < i_off else x_base

    v_s = float(bias)
    v_t = 0.0
    drive_lp = float(bias)
    x_hist = [x_base] * (gate_lag + 2)

    for _ in range(n_settle):
        x_t = x_base
        x_d = x_hist[-1 - gate_lag]
        _, v_s, v_t, drive_lp, _ = update_step(
            v_s, v_t, drive_lp, x_t, x_d,
            bias=bias, adapt_gain=adapt_gain, tau_m=tau_m, tau_adapt=tau_adapt,
            gate_pivot=gate_pivot, dt=dt,
        )
        x_hist.append(x_t)
        x_hist.pop(0)

    t = np.arange(n) * dt
    a_trace = np.empty(n)
    vs_trace = np.empty(n)
    vt_trace = np.empty(n)
    r_trace = np.empty(n)
    g_trace = np.empty(n)
    for i in range(n):
        x_t = x_at(i)
        x_d = x_hist[-1 - gate_lag] if len(x_hist) > gate_lag else x_base
        a, v_s, v_t, drive_lp, gate = update_step(
            v_s, v_t, drive_lp, x_t, x_d,
            bias=bias, adapt_gain=adapt_gain, tau_m=tau_m, tau_adapt=tau_adapt,
            gate_pivot=gate_pivot, dt=dt,
        )
        x_hist.append(x_t)
        x_hist.pop(0)
        a_trace[i] = a
        vs_trace[i] = v_s
        vt_trace[i] = v_t
        r_trace[i] = drive_lp
        g_trace[i] = gate
    return t, a_trace, vs_trace, vt_trace, r_trace, g_trace


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--X-lo", type=float, default=0.0, help="drive axis low for gate panel")
    p.add_argument("--X-hi", type=float, default=1.0, help="drive axis high for gate panel")
    p.add_argument("--n", type=int, default=401, help="number of X samples for gate panel")
    p.add_argument("--bias", type=float, default=None)
    p.add_argument("--tau-m", type=float, default=None, help="time constant [ms]")
    p.add_argument("--gate-pivot", type=float, default=float(fc.GATE_PIVOT))
    p.add_argument(
        "--gadapt-list",
        type=str,
        default=DEFAULT_GADAPT_LIST,
        help="comma-separated adapt_gain values for pulse panel",
    )
    p.add_argument(
        "--tau-adapt-list",
        type=str,
        default=DEFAULT_TAU_ADAPT_LIST,
        help="comma-separated tau_adapt [ms] for tau-adapt pulse panel",
    )
    p.add_argument(
        "--fixed-gadapt",
        type=float,
        default=DEFAULT_FIXED_GADAPT,
        help="fixed adapt_gain for tau_adapt sweep panels",
    )
    p.add_argument(
        "--fixed-tau-adapt",
        type=float,
        default=DEFAULT_FIXED_TAU_ADAPT,
        help="fixed tau_adapt [ms] for adapt_gain sweep panels",
    )
    p.add_argument("--x-base", type=float, default=0.25, help="baseline normalized input x_t")
    p.add_argument("--x-pulse", type=float, default=0.75, help="pulse normalized input x_t")
    p.add_argument("--t-total", type=float, default=2000.0, help="pulse sim length [ms]")
    p.add_argument("--t-pulse-on", type=float, default=200.0, help="pulse start [ms]")
    p.add_argument("--t-pulse-off", type=float, default=1000.0, help="pulse end [ms]")
    p.add_argument("--settle", type=float, default=5000.0, help="pre-pulse settle [ms]")
    p.add_argument("--save", type=str, default=DEFAULT_SAVE)
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    bias = args.bias if args.bias is not None else _init("bias")
    tau_m = args.tau_m if args.tau_m is not None else _init("tau_m")
    gate_pivot = args.gate_pivot
    gadapt_list = [float(x) for x in parse_comma_list(args.gadapt_list)]
    tau_adapt_list = [float(x) for x in parse_comma_list(args.tau_adapt_list)]
    g_fix = args.fixed_gadapt
    tau_fix = args.fixed_tau_adapt

    X_axis = np.linspace(args.X_lo, args.X_hi, args.n)
    fig = plt.figure(figsize=(14.0, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_gate = fig.add_subplot(gs[:, 0])
    ax_a_g = fig.add_subplot(gs[0, 1])
    ax_vt_g = fig.add_subplot(gs[1, 1])
    ax_a_tau = fig.add_subplot(gs[0, 2])
    ax_vt_tau = fig.add_subplot(gs[1, 2])

    cmap_g = plt.cm.viridis(np.linspace(0.15, 0.9, len(gadapt_list)))
    for g_ad, color in zip(gadapt_list, cmap_g):
        ax_gate.plot(
            X_axis,
            contrast_gate(X_axis, g_ad, gate_pivot),
            color=color,
            label=rf"$g_{{\mathrm{{adapt}}}}$={g_ad:g}",
        )
    ax_gate.axvline(gate_pivot, color="0.5", ls=":", lw=0.8)
    ax_gate.axhline(0.0, color="k", lw=0.6)
    ax_gate.set_xlabel(r"$X_{\mathrm{gate}}$")
    ax_gate.set_ylabel(r"$g=(X_{\mathrm{gate}}-\mathrm{pivot})\,g_{\mathrm{adapt}}$")
    ax_gate.set_title(rf"contrast gate (pivot={gate_pivot:g})")
    ax_gate.legend(fontsize=8, loc="best")
    ax_gate.grid(True, alpha=0.3)

    common_kwargs = dict(
        bias=bias,
        tau_m=tau_m,
        gate_pivot=gate_pivot,
        x_base=args.x_base,
        x_pulse=args.x_pulse,
        t_total_ms=args.t_total,
        t_pulse_on_ms=args.t_pulse_on,
        t_pulse_off_ms=args.t_pulse_off,
        settle_ms=args.settle,
    )

    for g_ad, color in zip(gadapt_list, cmap_g):
        t, a, _vs, vt, _r, _g = simulate_pulse(
            g_ad, tau_adapt=tau_fix, **common_kwargs,
        )
        lab = rf"$g_{{\mathrm{{adapt}}}}$={g_ad:g}"
        ax_a_g.plot(t, a, color=color, lw=1.5, label=lab)
        ax_vt_g.plot(t, vt, color=color, lw=1.5, label=lab)

    cmap_tau = plt.cm.plasma(np.linspace(0.15, 0.85, len(tau_adapt_list)))
    for tau_ad, color in zip(tau_adapt_list, cmap_tau):
        t, a, _vs, vt, _r, _g = simulate_pulse(
            g_fix, tau_adapt=tau_ad, **common_kwargs,
        )
        lab = rf"$\tau_{{\mathrm{{adapt}}}}$={tau_ad:g}"
        ax_a_tau.plot(t, a, color=color, lw=1.5, label=lab)
        ax_vt_tau.plot(t, vt, color=color, lw=1.5, label=lab)

    pulse_axes = (ax_a_g, ax_vt_g, ax_a_tau, ax_vt_tau)
    for ax in pulse_axes:
        ax.axvspan(args.t_pulse_on, args.t_pulse_off, color="0.85", zorder=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    ax_a_g.axhline(bias, color="k", ls=":", lw=0.8)
    ax_a_g.set_ylabel(r"$a$")
    ax_a_g.set_title(
        rf"sweep $g_{{\mathrm{{adapt}}}}$, "
        rf"$\tau_{{\mathrm{{adapt}}}}$={tau_fix:g} ms, $\tau_m$={tau_m:g} ms"
    )
    ax_vt_g.set_xlabel(r"$t$ [ms]")
    ax_vt_g.set_ylabel(r"$v_{\mathrm{transient}}$")

    ax_a_tau.axhline(bias, color="k", ls=":", lw=0.8)
    ax_a_tau.set_ylabel(r"$a$")
    ax_a_tau.set_title(
        rf"sweep $\tau_{{\mathrm{{adapt}}}}$, "
        rf"$g_{{\mathrm{{adapt}}}}$={g_fix:g}, $\tau_m$={tau_m:g} ms"
    )
    ax_vt_tau.set_xlabel(r"$t$ [ms]")
    ax_vt_tau.set_ylabel(r"$v_{\mathrm{transient}}$")

    fig.suptitle(
        rf"$x$={args.x_base:g}$\rightarrow${args.x_pulse:g}, "
        rf"bias={bias:g}, $\Delta t$={DELTA_MS:g} ms, lag={fc.gate_lag}",
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
