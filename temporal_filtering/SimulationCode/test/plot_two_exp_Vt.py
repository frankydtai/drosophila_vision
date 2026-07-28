"""Plot two-exponential membrane potential with a step input window.

Step input:
    S(t) = 0                          for t < t_pulse_on
    S(t) = S0                         for t_pulse_on <= t < t_pulse_off
    S(t) = 0                          for t >= t_pulse_off

Two-exponential transient response (difference of exponentials):

    V(t) = Vrest + S(t) convolved with a two-exp kernel

Closed-form using superposition of a +S0 step at ``t_pulse_on`` and a -S0 step at
``t_pulse_off``:

    V(t) = Vrest
           + S0*(Ks*exp(-(t-t_on)/tau_s) - Kf*exp(-(t-t_on)/tau_f)) * 1_{t>=t_on}
           - S0*(Ks*exp(-(t-t_off)/tau_s) - Kf*exp(-(t-t_off)/tau_f)) * 1_{t>=t_off}

Example:
    ../.venv/bin/python test/plot_two_exp_Vt.py --show
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Ensure we can import parse_comma_list without relying on CWD/pythonpath.
TFROOT = os.path.dirname(os.path.dirname(HERE))  # temporal_filtering/
sys.path.insert(0, os.path.join(TFROOT, "Connectome", "FAFBv783"))

from connectome_io import parse_comma_list  # type: ignore


def V_two_exp_step_window(
    t: np.ndarray,
    *,
    Vrest: float,
    Ks: float,
    Kf: float,
    S0: float,
    tau_f: float,
    tau_s: float,
    t_on: float,
    t_off: float,
) -> np.ndarray:
    """Closed-form for a +S0 step at t_on and a -S0 step at t_off."""
    t_rel_on = t - t_on
    t_rel_off = t - t_off
    on_mask = (t_rel_on >= 0.0).astype(np.float64)
    off_mask = (t_rel_off >= 0.0).astype(np.float64)

    term_on = (Ks * np.exp(-t_rel_on / tau_s) - Kf * np.exp(-t_rel_on / tau_f)) * on_mask
    term_off = (Ks * np.exp(-t_rel_off / tau_s) - Kf * np.exp(-t_rel_off / tau_f)) * off_mask
    return Vrest + S0 * (term_on - term_off)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Vrest", type=float, default=-65.0, help="Vrest")
    p.add_argument("--Ks", type=float, default=1.0, help="Ks (slow exponential gain)")
    p.add_argument("--Kf", type=float, default=1.0, help="Kf (fast exponential gain)")
    p.add_argument("--S0", type=float, default=1.0, help="S0")
    p.add_argument("--Ks-list", type=str, default="0,0.5,1.0,2.0", help="comma-separated Ks sweep values")
    p.add_argument("--Kf-list", type=str, default="0,0.5,1.0,2.0", help="comma-separated Kf sweep values")
    p.add_argument("--tau-s-list", type=str, default="100,200,400", help="comma-separated tau_s sweep values")
    p.add_argument("--tau-f-list", type=str, default="20,50,100", help="comma-separated tau_f sweep values")
    p.add_argument("--tau-f", type=float, default=50.0, help="tau_f (fast) > 0")
    p.add_argument("--tau-s", type=float, default=200.0, help="tau_s (slow) > 0")
    p.add_argument("--t-total", type=float, default=2000.0, help="pulse sim length")
    p.add_argument("--t-pulse-on", type=float, default=200.0, help="pulse start time")
    p.add_argument("--t-pulse-off", type=float, default=1000.0, help="pulse end time")
    p.add_argument("--dt", type=float, default=0.1, help="time step")
    p.add_argument("--logx", action="store_true", help="use log-x axis")
    default_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_exp_Vt.png")
    p.add_argument("--save", type=str, default=default_save, help="output PNG path")
    p.add_argument("--show", action="store_true", help="show interactive plot")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.tau_f <= 0 or args.tau_s <= 0:
        raise ValueError("tau-f and tau-s must be > 0")
    if args.dt <= 0:
        raise ValueError("--dt must be > 0")
    if args.t_total <= 0:
        raise ValueError("--t-total must be > 0")
    if args.t_pulse_on < 0 or args.t_pulse_off < 0:
        raise ValueError("pulse times must be >= 0")
    if args.t_pulse_off <= args.t_pulse_on:
        raise ValueError("--t-pulse-off must be > --t-pulse-on")

    n = int(round(args.t_total / args.dt))
    t = np.arange(n, dtype=np.float64) * args.dt
    i_on = int(round(args.t_pulse_on / args.dt))
    i_off = int(round(args.t_pulse_off / args.dt))

    S = np.zeros_like(t)
    S[i_on:i_off] = args.S0

    Ks_values = [float(x) for x in parse_comma_list(args.Ks_list)]
    Kf_values = [float(x) for x in parse_comma_list(args.Kf_list)]
    tau_s_values = [float(x) for x in parse_comma_list(args.tau_s_list)]
    tau_f_values = [float(x) for x in parse_comma_list(args.tau_f_list)]

    n_cols = max(len(Kf_values), len(tau_f_values))
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(4.2 * n_cols, 7.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes).reshape(2, n_cols)

    cmap = plt.cm.viridis
    all_traces = []

    def _shade_pulse(ax):
        ax.axvspan(args.t_pulse_on, args.t_pulse_off, color="0.85", zorder=0)

    # top row: each column is one Kf, traces sweep Ks (tau_f fixed)
    for col, Kf_val in enumerate(Kf_values):
        ax = axes[0, col]
        for i, Ks_val in enumerate(Ks_values):
            color = cmap(0.15 + 0.75 * (i / max(1, len(Ks_values) - 1)))
            Vc = V_two_exp_step_window(
                t,
                Vrest=args.Vrest,
                Ks=Ks_val,
                Kf=Kf_val,
                S0=args.S0,
                tau_f=args.tau_f,
                tau_s=args.tau_s,
                t_on=args.t_pulse_on,
                t_off=args.t_pulse_off,
            )
            all_traces.append(Vc)
            ax.plot(t, Vc, color=color, lw=2.0, label=rf"$K_s$={Ks_val:g}")

        _shade_pulse(ax)
        ax.axhline(args.Vrest, color="k", ls=":", lw=1.0)
        ax.set_title(rf"$K_f$={Kf_val:g}$,\ \tau_f$={args.tau_f:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if col == 0:
            ax.set_ylabel("V(t)")

    # bottom row: each column is one tau_f, traces sweep tau_s (Kf fixed)
    for col, tau_f_val in enumerate(tau_f_values):
        ax = axes[1, col]
        for i, tau_s_val in enumerate(tau_s_values):
            color = cmap(0.15 + 0.75 * (i / max(1, len(tau_s_values) - 1)))
            Vc = V_two_exp_step_window(
                t,
                Vrest=args.Vrest,
                Ks=args.Ks,
                Kf=args.Kf,
                S0=args.S0,
                tau_f=tau_f_val,
                tau_s=tau_s_val,
                t_on=args.t_pulse_on,
                t_off=args.t_pulse_off,
            )
            all_traces.append(Vc)
            ax.plot(t, Vc, color=color, lw=2.0, label=rf"$\tau_s$={tau_s_val:g}")

        _shade_pulse(ax)
        ax.axhline(args.Vrest, color="k", ls=":", lw=1.0)
        ax.set_title(rf"$\tau_f$={tau_f_val:g}$,\ K_f$={args.Kf:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        if col == 0:
            ax.set_ylabel("V(t)")
        ax.set_xlabel("t")

    # hide unused cells if the two column counts differ
    for col in range(len(Kf_values), n_cols):
        axes[0, col].set_visible(False)
    for col in range(len(tau_f_values), n_cols):
        axes[1, col].set_visible(False)

    fig.suptitle(
        rf"top row: $K_f$ vs $K_s$ (fixed $\tau_f={args.tau_f:g}$, $\tau_s={args.tau_s:g}$)"
        + "\n"
        + rf"bottom row: $\tau_f$ vs $\tau_s$ (fixed $K_f={args.Kf:g}$, $K_s={args.Ks:g}$), $V(t)=V_{{rest}}+(S*h)(t)$",
        fontsize=11,
    )

    ymin = min(float(np.min(v)) for v in all_traces)
    ymax = max(float(np.max(v)) for v in all_traces)
    pad = max(0.05 * (ymax - ymin), 1.0e-3)
    for ax in axes.ravel():
        ax.set_ylim(ymin - pad, ymax + pad)

    if args.logx:
        for ax in axes.ravel():
            ax.set_xscale("log")

    out_path = args.save
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

