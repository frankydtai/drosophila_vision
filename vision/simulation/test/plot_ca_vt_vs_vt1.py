"""Compare Ca low-pass drive at v[t] vs v[t-1] on a pulse v_delta.

Current path (``filter_ca`` / ``forward_full``):

    ca[t] = ca[t-1] + (Δt/τ_ca) ((v[t] - v_ref) - ca[t-1])

Alternate (one-step lag on the drive):

    ca[t] = ca[t-1] + (Δt/τ_ca) ((v[t-1] - v_ref) - ca[t-1])

Usage (from ``vision/simulation/``):

    ../.venv/bin/python test/plot_ca_vt_vs_vt1.py
    ../.venv/bin/python test/plot_ca_vt_vs_vt1.py --show
    ../.venv/bin/python test/plot_ca_vt_vs_vt1.py --pulse-list 50,100,500
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

from import_bootstrap import parse_comma_list
from figure.panel import TRACE_LW, save_figure
from neuron.filter_ca import filter_ca
from const_default import NEURON_CONST, NEURON_SCHEMA

DEFAULT_SAVE = os.path.join(HERE, "ca_vt_vs_vt1.png")
DEFAULT_PULSE_LIST = "50,100,500"
DEFAULT_DELTA_MS = float(NEURON_CONST["delta_ms"]["ca"])
DEFAULT_TAU_CA = float(NEURON_SCHEMA["params"]["tau_ca"]["init"])
V_AMP = 20.0
T_ON_MS = 500.0
T_END_MS = 2000.0


def _v_pulse(n: int, t_on: int, pulse_t: int, amp: float) -> np.ndarray:
    v = np.zeros(n, dtype=np.float64)
    t_off = min(n, t_on + pulse_t)
    v[t_on:t_off] = amp
    return v


def _ca_from_v(
    v_delta: np.ndarray, *, use_v_prev: bool, t_on: int, delta_ms: float, tau_ca: float,
) -> np.ndarray:
    """Forward Ca on ``v_delta`` (= v - v_ref). ``t_on`` resets ca to 0."""
    ca = np.zeros_like(v_delta)
    prev = 0.0
    for t in range(1, v_delta.shape[0]):
        if t == t_on:
            prev = 0.0
        drive = v_delta[t - 1] if use_v_prev else v_delta[t]
        prev = filter_ca(prev, drive, delta_ms=delta_ms, tau_ca=tau_ca)
        ca[t] = prev
    return ca


def _plot(pulse_mss, dt_ms, tau_ca, save, show):
    n_pulse = len(pulse_mss)
    fig, axes = plt.subplots(
        3, n_pulse, figsize=(3.2 * n_pulse, 7.2), squeeze=False, sharex="col",
    )
    t_on = int(round(T_ON_MS / dt_ms))
    n = int(round(T_END_MS / dt_ms))
    t_s = np.arange(n) * dt_ms / 1000.0
    dt_over_tau_ca = dt_ms / tau_ca

    for col, ms_pulse in enumerate(pulse_mss):
        pulse_t = max(1, int(round(float(ms_pulse) / dt_ms)))
        v = _v_pulse(n, t_on, pulse_t, V_AMP)
        ca_t = _ca_from_v(
            v, use_v_prev=False, t_on=t_on, delta_ms=dt_ms, tau_ca=tau_ca,
        )
        ca_tm1 = _ca_from_v(
            v, use_v_prev=True, t_on=t_on, delta_ms=dt_ms, tau_ca=tau_ca,
        )
        d = ca_t - ca_tm1

        ax0, ax1, ax2 = axes[0][col], axes[1][col], axes[2][col]
        ax0.plot(t_s, v, color="0.35", lw=TRACE_LW, label="v − v_ref")
        ax0.set_title(f"pulse {ms_pulse:g} ms", fontsize=10)
        ax0.set_ylabel("mV", fontsize=8)
        ax0.axhline(0.0, color="0.8", lw=0.5)
        ax0.tick_params(labelsize=7)

        ax1.plot(t_s, ca_t, color="C0", lw=TRACE_LW, label="ca ← v[t]")
        ax1.plot(t_s, ca_tm1, color="C1", lw=TRACE_LW, label="ca ← v[t−1]")
        ax1.set_ylabel("ca", fontsize=8)
        ax1.axhline(0.0, color="0.8", lw=0.5)
        ax1.tick_params(labelsize=7)

        ax2.plot(t_s, d, color="C3", lw=TRACE_LW, label="ca(v[t]) − ca(v[t−1])")
        ax2.set_ylabel("Δ ca", fontsize=8)
        ax2.set_xlabel("t (s)", fontsize=8)
        ax2.axhline(0.0, color="0.8", lw=0.5)
        ax2.tick_params(labelsize=7)

        peak = float(np.max(np.abs(d)))
        ax2.text(
            0.98, 0.95, f"|Δ|_max={peak:.4g}",
            transform=ax2.transAxes, ha="right", va="top", fontsize=7,
        )

    handles0, labels0 = axes[0][0].get_legend_handles_labels()
    handles1, labels1 = axes[1][0].get_legend_handles_labels()
    handles2, labels2 = axes[2][0].get_legend_handles_labels()
    fig.legend(
        handles0 + handles1 + handles2,
        labels0 + labels1 + labels2,
        loc="upper right",
        fontsize=8,
    )
    fig.suptitle(
        rf"Ca drive: v[t] vs v[t−1]  "
        rf"($\Delta t$={dt_ms:g} ms, $\tau_{{\mathrm{{ca}}}}$={tau_ca:g} ms, "
        rf"$\Delta t/\tau_{{\mathrm{{ca}}}}$={dt_over_tau_ca:g})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save)
    print(f"saved {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--pulse-list", default=DEFAULT_PULSE_LIST)
    ap.add_argument("--delta-ms", type=float, default=DEFAULT_DELTA_MS)
    ap.add_argument("--tau-ca", type=float, default=DEFAULT_TAU_CA)
    args = ap.parse_args()

    pulse_mss = [float(x) for x in parse_comma_list(args.pulse_list)]
    if not pulse_mss:
        raise SystemExit("empty --pulse-list")
    if args.delta_ms <= 0:
        raise SystemExit("--delta-ms must be > 0")
    if args.tau_ca <= 0:
        raise SystemExit("--tau-ca must be > 0")
    _plot(pulse_mss, float(args.delta_ms), float(args.tau_ca), args.save, args.show)


if __name__ == "__main__":
    main()
