"""Arenz Table S1/S2 control: freq H + time h(t)=A_HP+B_LP.

From H(s)=A s τ_HP / ((1+s τ_HP)(1+s τ_LP)), partial fractions give
  h(t)=A/(τ_HP-τ_LP) · [(τ_HP/τ_LP)e^{-t/τ_LP} - e^{-t/τ_HP}]
  A_HP = -e^{-t/τ_HP}/(τ_HP-τ_LP)          always ≤ 0
  B_LP = (τ_HP/τ_LP)e^{-t/τ_LP}/(τ_HP-τ_LP) always ≥ 0
  h    = A_HP + B_LP

LP-only: h(t)=B_LP = e^{-t/τ_LP}/τ_LP  (no A_HP).

Bottom 8 panels (freq): H_raw, H_deconv (solid), H_deconv·H_GCaMP, H_raw/H_GCaMP.

Run:  ../.venv/bin/python arenz_h.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_PNG = HERE / "arenz_h.png"

F_MIN, F_MAX = 0.01, 5.5
T_MAX_S = 4.0
TAU_GCAMP_S = 0.350  # Arenz GCaMP6f deconv LP
# (cell, pathway, kind, tau_hp_raw_s, tau_lp_raw_s, tau_hp_deconv_s, tau_lp_deconv_s)
CELLS = [
    ("Mi1", "ON", "bp", 1.078, 0.266, 0.318, 0.054),
    ("Tm3", "ON", "bp", 1.769, 0.158, 0.260, 0.027),
    ("Mi4", "ON", "lp", None, 0.519, None, 0.038),
    ("Mi9", "ON", "lp", None, 0.546, None, 0.077),
    ("Tm1", "OFF", "bp", 0.632, 0.271, 0.296, 0.044),
    ("Tm2", "OFF", "bp", 0.962, 0.113, 0.153, 0.014),
    ("Tm4", "OFF", "bp", 0.788, 0.186, 0.249, 0.024),
    ("Tm9", "OFF", "lp", None, 0.462, None, 0.017),
]


def h_lp_freq(f: np.ndarray, tau_s: float) -> np.ndarray:
    w = 2.0 * np.pi * f * tau_s
    return 1.0 / (1.0 + 1j * w)


def h_hp_freq(f: np.ndarray, tau_s: float) -> np.ndarray:
    w = 2.0 * np.pi * f * tau_s
    return (1j * w) / (1.0 + 1j * w)


def h_bp_freq(f: np.ndarray, tau_hp_s: float, tau_lp_s: float) -> np.ndarray:
    return h_hp_freq(f, tau_hp_s) * h_lp_freq(f, tau_lp_s)


def mag_db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.abs(h) + 1e-30)


def h_bp_terms(t: np.ndarray, tau_hp_s: float, tau_lp_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A_HP (negative) + B_LP (positive) = h; matches IFFT of H_HP·H_LP."""
    denom = tau_hp_s - tau_lp_s
    a_hp = -np.exp(-t / tau_hp_s) / denom
    b_lp = (tau_hp_s / tau_lp_s) * np.exp(-t / tau_lp_s) / denom
    h = a_hp + b_lp
    return a_hp, b_lp, h


def b_lp_only(t: np.ndarray, tau_lp_s: float) -> np.ndarray:
    """Pure LP impulse: h = B_LP = e^{-t/τ}/τ."""
    return np.exp(-t / tau_lp_s) / tau_lp_s


def h_cell_freq(kind: str, f: np.ndarray, tau_hp_s: float | None, tau_lp_s: float) -> np.ndarray:
    if kind == "bp":
        return h_bp_freq(f, tau_hp_s, tau_lp_s)
    return h_lp_freq(f, tau_lp_s)


def h_bp_ifft(t: np.ndarray, tau_hp_s: float, tau_lp_s: float) -> np.ndarray:
    dt = float(t[1] - t[0])
    f = np.fft.rfftfreq(len(t), d=dt)
    return np.fft.irfft(h_bp_freq(f, tau_hp_s, tau_lp_s), n=len(t)) / dt


def verify_formulas(t: np.ndarray) -> None:
    for tau_hp_s, tau_lp_s in ((1.078, 0.266), (0.632, 0.271)):
        a, b, h = h_bp_terms(t, tau_hp_s, tau_lp_s)
        if not np.allclose(h, a + b):
            raise AssertionError("h != A_HP + B_LP")
        if np.any(a > 1e-12) or np.any(b < -1e-12):
            raise AssertionError("sign: A_HP must be ≤0, B_LP must be ≥0")
        h_ref = h_bp_ifft(t, tau_hp_s, tau_lp_s)
        if np.corrcoef(h, h_ref)[0, 1] < 0.99:
            raise AssertionError(f"h(t) vs IFFT mismatch for τ_HP={tau_hp_s} τ_LP={tau_lp_s}")


def tau_box(ax: plt.Axes, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv) -> None:
    hp_raw_ms = "-" if tau_hp_raw is None else f"{tau_hp_raw * 1000:.0f}"
    hp_deconv_ms = "-" if tau_hp_deconv is None else f"{tau_hp_deconv * 1000:.0f}"
    tau_text = (
        rf"$\tau_{{LP,raw}}$={tau_lp_raw * 1000:.0f} ms" "\n"
        rf"$\tau_{{HP,raw}}$={hp_raw_ms} ms" "\n"
        rf"$\tau_{{LP,deconv}}$={tau_lp_deconv * 1000:.0f} ms" "\n"
        rf"$\tau_{{HP,deconv}}$={hp_deconv_ms} ms"
    )
    ax.text(
        0.03,
        0.97,
        tau_text,
        transform=ax.transAxes,
        va="top",
        fontsize=6,
        color="0.2",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="0.8"),
    )


def plot_freq(ax: plt.Axes, kind: str, freqs: np.ndarray, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv) -> None:
    lp_raw = h_lp_freq(freqs, tau_lp_raw)
    lp_deconv = h_lp_freq(freqs, tau_lp_deconv)
    hp_raw = h_hp_freq(freqs, tau_hp_raw) if kind == "bp" else np.ones_like(freqs)
    hp_deconv = h_hp_freq(freqs, tau_hp_deconv) if kind == "bp" else np.ones_like(freqs)
    for h, label, color, lw in zip(
        (lp_raw, hp_raw),
        (r"$H_{\mathrm{LP,raw}}$", r"$H_{\mathrm{HP,raw}}$"),
        ("C0", "C1"),
        (1.1, 1.4),
    ):
        ax.plot(freqs, mag_db(h), lw=lw, color=color, label=label)
    for h, label, color, lw in zip(
        (lp_deconv, hp_deconv),
        (r"$H_{\mathrm{LP,deconv}}$", r"$H_{\mathrm{HP,deconv}}$"),
        ("C0", "C1"),
        (1.1, 1.4),
    ):
        ax.plot(freqs, mag_db(h), lw=lw, ls="--", color=color, alpha=0.9, label=label)
    ax.set_xscale("log")
    ax.set_ylabel("gain (dB)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)


def plot_time(ax: plt.Axes, kind: str, t: np.ndarray, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv) -> None:
    if kind == "bp":
        a_r, b_r, h_r = h_bp_terms(t, tau_hp_raw, tau_lp_raw)
        a_d, b_d, h_d = h_bp_terms(t, tau_hp_deconv, tau_lp_deconv)
        ax.plot(t, a_r, color="C1", lw=1.1, label=r"$A_{\mathrm{HP}}$ (raw)")
        ax.plot(t, b_r, color="C0", lw=1.1, label=r"$B_{\mathrm{LP}}$ (raw)")
        ax.plot(t, a_d, color="C1", lw=1.1, ls="--", alpha=0.9, label=r"$A_{\mathrm{HP}}$ (deconv)")
        ax.plot(t, b_d, color="C0", lw=1.1, ls="--", alpha=0.9, label=r"$B_{\mathrm{LP}}$ (deconv)")
        ax.plot(t, h_r, color="C2", lw=2.0, label=r"$A_{\mathrm{HP}}+B_{\mathrm{LP}}$ raw")
        ax.plot(t, h_d, color="C2", lw=2.0, ls="--", alpha=0.9, label=r"$A_{\mathrm{HP}}+B_{\mathrm{LP}}$ deconv")
        # ylim: exclude blue dashed B_LP deconv (can spike at t=0)
        lim_data = np.concatenate([a_r, a_d, b_r, h_r])
    else:
        b_r = b_lp_only(t, tau_lp_raw)
        b_d = b_lp_only(t, tau_lp_deconv)
        ax.plot(t, b_r, color="C0", lw=1.1, label=r"$B_{\mathrm{LP}}$ (raw)")
        ax.plot(t, b_d, color="C0", lw=1.1, ls="--", alpha=0.9, label=r"$B_{\mathrm{LP}}$ (deconv)")
        lim_data = b_r
    lo, hi = float(np.min(lim_data)), float(np.max(lim_data))
    pad = 0.1 * (hi - lo if hi > lo else max(abs(lo), abs(hi), 1.0))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel(r"$h$ term", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="0.75", lw=0.5)


def plot_h_deconv_check(
    ax: plt.Axes,
    kind: str,
    freqs: np.ndarray,
    tau_hp_raw,
    tau_lp_raw,
    tau_hp_deconv,
    tau_lp_deconv,
) -> None:
    h_gcamp = h_lp_freq(freqs, TAU_GCAMP_S)
    h_raw = h_cell_freq(kind, freqs, tau_hp_raw, tau_lp_raw)
    h_deconv = h_cell_freq(kind, freqs, tau_hp_deconv, tau_lp_deconv)
    h_recon = h_deconv * h_gcamp
    h_raw_over_gcamp = h_raw / h_gcamp
    ax.plot(freqs, mag_db(h_raw), color="C2", lw=1.8, label=r"$H_{\mathrm{raw}}$")
    ax.plot(freqs, mag_db(h_deconv), color="C3", lw=1.4, label=r"$H_{\mathrm{deconv}}$")
    ax.plot(
        freqs,
        mag_db(h_recon),
        color="C4",
        lw=1.4,
        ls=":",
        label=r"$H_{\mathrm{deconv}}\,H_{\mathrm{GCaMP}}$",
    )
    ax.plot(
        freqs,
        mag_db(h_raw_over_gcamp),
        color="C5",
        lw=1.4,
        ls="--",
        label=r"$H_{\mathrm{raw}}/H_{\mathrm{GCaMP}}$",
    )
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)", fontsize=8)
    ax.set_ylabel("gain (dB)", fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=5.5, loc="best", framealpha=0.85)


def main() -> int:
    t = np.linspace(0.0, T_MAX_S, 800)
    verify_formulas(t)
    freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), 400)

    fig, axes = plt.subplots(6, 4, figsize=(12, 19))
    for col, row in enumerate(CELLS[:4]):
        cell, pathway, kind, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv = row
        ax_f = axes[0, col]
        ax_f.set_title(f"{cell} ({pathway})", fontsize=9)
        plot_freq(ax_f, kind, freqs, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        tau_box(ax_f, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        axes[2, col].set_title(f"{cell} — $h(t)$ terms", fontsize=9)
        plot_time(axes[2, col], kind, t, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        axes[4, col].set_title(f"{cell} — $H$ deconv check", fontsize=9)
        plot_h_deconv_check(
            axes[4, col], kind, freqs, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv
        )

    for col, row in enumerate(CELLS[4:]):
        cell, pathway, kind, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv = row
        ax_f = axes[1, col]
        ax_f.set_title(f"{cell} ({pathway})", fontsize=9)
        plot_freq(ax_f, kind, freqs, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        tau_box(ax_f, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        axes[3, col].set_title(f"{cell} — $h(t)$ terms", fontsize=9)
        plot_time(axes[3, col], kind, t, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv)
        axes[5, col].set_title(f"{cell} — $H$ deconv check", fontsize=9)
        plot_h_deconv_check(
            axes[5, col], kind, freqs, tau_hp_raw, tau_lp_raw, tau_hp_deconv, tau_lp_deconv
        )

    axes[0, 0].set_ylabel("gain (dB)\nfreq", fontsize=8)
    axes[2, 0].set_ylabel(r"$h$ term" + "\ntime", fontsize=8)
    axes[4, 0].set_ylabel("gain (dB)\ndeconv", fontsize=8)
    axes[0, 0].legend(fontsize=5.5, loc="lower left", framealpha=0.85)
    axes[2, 0].legend(fontsize=5.5, loc="upper right", framealpha=0.85)
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    for ax in axes[1]:
        ax.tick_params(labelbottom=False)
    for ax in axes[2]:
        ax.tick_params(labelbottom=False)
    for ax in axes[4]:
        ax.tick_params(labelbottom=False)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.03, top=0.96, hspace=0.50, wspace=0.32)
    fig.suptitle(
        r"$H_{\mathrm{raw}} \approx H_{\mathrm{deconv}}\,H_{\mathrm{GCaMP}}$"
        rf" ($\tau_{{\mathrm{{GCaMP}}}}={TAU_GCAMP_S * 1000:.0f}$ ms); "
        r"$h_{\mathrm{BP}}=\frac{\frac{\tau_{HP}}{\tau_{LP}}e^{-t/\tau_{LP}}-e^{-t/\tau_{HP}}}{\tau_{HP}-\tau_{LP}}$"
        " (Arenz control, Table S1/S2)",
        y=0.995,
        fontsize=9,
    )
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
