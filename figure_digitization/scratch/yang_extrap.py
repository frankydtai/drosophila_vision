"""Yang S5 digitized V — tail smooth compare + peak decay extrap.

Outputs:
  yang_extrap_full.png      filter on entire digitized span
  yang_extrap_tail450.png   tail 450 ms smooth only
  yang_extrap.png           tail smooth + peak decay fit grid

Run:  ../.venv/bin/python yang_extrap.py
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, savgol_filter

HERE = Path(__file__).resolve().parent
YANG_CSV = HERE.parent / "yang" / "s5be_digitized.csv"
OUT_FULL = HERE / "yang_extrap_full.png"
OUT_TAIL450 = HERE / "yang_extrap_tail450.png"
OUT_TAIL450_EXTRAP = HERE / "yang_extrap.png"

DT = 1.0 / 230.0
TAIL450_MS = 450.0
TAU_V_MIN_S = 0.02
TAU_V_MAX_S = 0.8
EPS_FRAC = 0.05
ORDER_B = ["L1", "Tm3", "Mi1"]
ORDER_E = ["L2", "Tm1", "Tm2"]
COLORS = {
    "L1": "#2a5db0",
    "L2": "#2a5db0",
    "Tm3": "#d62728",
    "Tm1": "#d62728",
    "Mi1": "#2ca02c",
    "Tm2": "#2ca02c",
}

FILTERS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {}


def _savgol(win_s: float, poly: int = 3) -> Callable[[np.ndarray], np.ndarray]:
    def f(y: np.ndarray) -> np.ndarray:
        n = len(y)
        win = max(poly + 2 | 1, int(round(win_s / DT)) | 1)
        win = min(win, n if n % 2 == 1 else n - 1)
        return savgol_filter(y, win, poly, mode="interp")

    return f


def _butter(fc_hz: float, order: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    def f(y: np.ndarray) -> np.ndarray:
        b, a = butter(order, fc_hz, btype="low", fs=1.0 / DT)
        return filtfilt(b, a, y)

    return f


def _gauss(sigma_s: float) -> Callable[[np.ndarray], np.ndarray]:
    sigma = max(1.0, sigma_s / DT)

    def f(y: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(y, sigma, mode="nearest")

    return f


def _ema(tau_s: float) -> Callable[[np.ndarray], np.ndarray]:
    a = DT / tau_s

    def f(y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = out[i - 1] + a * (y[i] - out[i - 1])
        return out

    return f


def _poly(deg: int) -> Callable[[np.ndarray], np.ndarray]:
    def f(y: np.ndarray) -> np.ndarray:
        x = np.linspace(-1.0, 1.0, len(y))
        c = np.polyfit(x, y, deg)
        return np.polyval(c, x)

    return f


FILTERS = {
    "SavGol 150 ms p3": ("#e41a1c", _savgol(0.150, 3)),
    "SavGol 80 ms p2": ("#ff7f00", _savgol(0.080, 2)),
    "Butter 5 Hz": ("#4daf4a", _butter(5.0)),
    "Butter 12 Hz": ("#984ea3", _butter(12.0)),
    "Gaussian 60 ms": ("#377eb8", _gauss(0.060)),
    "EMA τ=80 ms": ("#a65628", _ema(0.080)),
    "Poly deg 4": ("#000000", _poly(4)),
}


def resample(t: np.ndarray, y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    order = np.argsort(t)
    t, y = t[order], y[order]
    _, idx = np.unique(t, return_index=True)
    t, y = t[idx], y[idx]
    return np.interp(t_grid, t, y, left=np.nan, right=np.nan)


def baseline0(y: np.ndarray) -> np.ndarray:
    m = np.isfinite(y)
    i0 = int(np.flatnonzero(m)[0]) if not m[0] else 0
    return y - y[i0]


def apply_full(vm: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.isfinite(vm)
    y = vm[finite].astype(float)
    i0 = int(np.flatnonzero(finite)[0])
    out: dict[str, np.ndarray] = {}
    for name, (_, fn) in FILTERS.items():
        v = vm.copy()
        ys = fn(y)
        v[finite] = ys + (vm[i0] - ys[0])
        out[name] = v
    return out


def apply_tail_one(
    vm: np.ndarray, t_grid: np.ndarray, tail_s: float, fn: Callable[[np.ndarray], np.ndarray]
) -> tuple[np.ndarray, float]:
    finite = np.isfinite(vm)
    i_end = int(np.flatnonzero(finite)[-1])
    t0 = float(t_grid[i_end]) - tail_s
    tail = finite & (t_grid >= t0) & (t_grid <= t_grid[i_end])
    i_t0 = int(np.flatnonzero(tail)[0])
    v = vm.copy()
    ys = fn(vm[tail].astype(float))
    v[tail] = ys + (vm[i_t0] - ys[0])
    return v, t0


def build_final_v(
    vm: np.ndarray, t_grid: np.ndarray, tail_s: float = TAIL450_MS / 1000.0
) -> tuple[np.ndarray, dict]:
    """Raw before peak; Butter 5 Hz tail smooth + peak decay extrap thereafter."""
    vf, _ = apply_tail_one(vm, t_grid, tail_s, _butter(5.0))
    _, vd, meta = extrap_from_peak(vf, t_grid, tail_s)
    t_peak = meta["t_peak"]
    out = np.full_like(vm, np.nan)
    meas = np.isfinite(vm)
    out[meas & (t_grid < t_peak)] = vm[meas & (t_grid < t_peak)]
    dm = np.isfinite(vd)
    out[dm] = vd[dm]
    return out, meta


def apply_tail(vm: np.ndarray, t_grid: np.ndarray, tail_s: float) -> tuple[dict[str, np.ndarray], float]:
    finite = np.isfinite(vm)
    i_end = int(np.flatnonzero(finite)[-1])
    t0 = float(t_grid[i_end]) - tail_s
    tail = finite & (t_grid >= t0) & (t_grid <= t_grid[i_end])
    i_t0 = int(np.flatnonzero(tail)[0])
    y_raw = vm[tail].astype(float)
    out: dict[str, np.ndarray] = {}
    for name, (_, fn) in FILTERS.items():
        v = vm.copy()
        ys = fn(y_raw)
        v[tail] = ys + (vm[i_t0] - ys[0])
        out[name] = v
    return out, t0


def fit_tau_log(tt: np.ndarray, vv: np.ndarray) -> float | None:
    m = np.abs(vv) > 1e-6
    if int(m.sum()) < 4:
        return None
    t = tt[m] - tt[m][0]
    slope = float(np.polyfit(t, np.log(np.abs(vv[m])), 1)[0])
    if slope >= 0:
        return None
    return float(np.clip(-1.0 / slope, TAU_V_MIN_S, TAU_V_MAX_S))


def extrap_from_peak(
    v: np.ndarray, t_grid: np.ndarray, tail_s: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Find |V| peak in last tail_s; fit exp decay from peak; extend to end of t_grid."""
    finite = np.isfinite(v)
    i_end = int(np.flatnonzero(finite)[-1])
    t_end = float(t_grid[i_end])
    t0 = t_end - tail_s
    tail = finite & (t_grid >= t0) & (t_grid <= t_end)
    idx_tail = np.flatnonzero(tail)
    i_pk = idx_tail[int(np.argmax(np.abs(v[tail])))]
    t_peak = float(t_grid[i_pk])
    v_peak = float(v[i_pk])

    fit = (t_grid >= t_peak) & (t_grid <= t_end)
    tt = t_grid[fit]
    vv = v[fit].astype(float)
    tau_v = 0.08

    def model(t: np.ndarray, tau: float) -> np.ndarray:
        return v_peak * np.exp(-(t - t_peak) / tau)

    if len(tt) >= 4:
        try:
            (tau_v,), _ = curve_fit(
                model,
                tt,
                vv,
                p0=(0.08,),
                bounds=([1e-4], [2.0]),
                maxfev=8000,
            )
            tau_v = float(tau_v)
        except RuntimeError:
            tv = fit_tau_log(tt - t_peak, vv)
            if tv is not None:
                tau_v = tv

    decay_idx = t_grid >= t_peak
    v_decay = model(t_grid[decay_idx], tau_v)
    ref = max(abs(v_peak), float(np.max(np.abs(vv))))
    eps = EPS_FRAC * ref if ref > 1e-9 else 1e-6
    hit = np.flatnonzero(np.abs(v_decay) < eps)
    if len(hit):
        v_decay[hit[0] :] = 0.0

    out_decay = np.full_like(v, np.nan)
    out_decay[decay_idx] = v_decay
    meta = dict(t0=t0, t_end=t_end, t_peak=t_peak, tau_v_ms=tau_v * 1e3)
    return v, out_decay, meta


RAW_COL = "Raw (no smooth)"


def plot_grid(
    vdf: pd.DataFrame,
    t_grid: np.ndarray,
    mode: str,
    tail_s: float | None,
    out_png: Path,
    title: str,
) -> None:
    filter_names = (
        [RAW_COL] + list(FILTERS.keys()) if mode == "tail_extrap" else list(FILTERS.keys())
    )
    n_f = len(filter_names)
    pairs = [(cell, stim) for cell in ORDER_B + ORDER_E for stim in ("light", "dark")]
    n_r = len(pairs)

    fig, axes = plt.subplots(n_r, n_f, figsize=(2.4 * n_f, 1.05 * n_r), sharex=True, sharey="row")

    for ri, (cell, stimulus) in enumerate(pairs):
        sub = vdf[(vdf.cell == cell) & (vdf.stimulus == stimulus)]
        vm = baseline0(resample(sub.time_s.values, sub.dff_percent.values, t_grid))
        meas = np.isfinite(vm)
        color = COLORS[cell]
        t0 = None
        t_end = None
        decayed: dict[str, np.ndarray] = {}
        extrap_meta: dict[str, dict] = {}
        if mode == "full":
            filtered = apply_full(vm)
        elif mode == "tail_extrap":
            filtered, t0 = apply_tail(vm, t_grid, float(tail_s))
            for name, vf in filtered.items():
                _, vd, meta = extrap_from_peak(vf, t_grid, float(tail_s))
                decayed[name] = vd
                extrap_meta[name] = meta
            _, vd_raw, meta_raw = extrap_from_peak(vm, t_grid, float(tail_s))
            decayed[RAW_COL] = vd_raw
            extrap_meta[RAW_COL] = meta_raw
            filtered[RAW_COL] = vm.copy()
            t_end = float(t_grid[int(np.flatnonzero(np.isfinite(vm))[-1])])
        else:
            filtered, t0 = apply_tail(vm, t_grid, float(tail_s))

        for fi, name in enumerate(filter_names):
            ax = axes[ri, fi]
            vf = filtered[name]
            if not (mode == "tail_extrap" and name == RAW_COL):
                ax.plot(t_grid[meas], vm[meas], color="0.8", alpha=0.45, lw=0.6)
            smooth_m = np.isfinite(vf)
            ax.plot(t_grid[smooth_m], vf[smooth_m], color=color, lw=1.2)
            if mode == "tail_extrap":
                vd = decayed[name]
                decay_m = np.isfinite(vd)
                ax.plot(t_grid[decay_m], vd[decay_m], color=color, lw=1.2, ls="--")
                ax.axvline(extrap_meta[name]["t_peak"], color="0.35", lw=0.5, ls="-.")
            if t0 is not None:
                ax.axvline(t0, color="0.6", lw=0.5, ls=":")
            if t_end is not None:
                ax.axvline(t_end, color="0.45", lw=0.5, ls="--")
            ax.axhline(0, color="0.85", lw=0.35)
            if ri == 0 and fi == n_f - 1:
                ax.plot([], [], color=color, lw=1.2, label="smooth")
                if mode == "tail_extrap":
                    ax.plot([], [], color=color, lw=1.2, ls="--", label="decay fit")
                    ax.legend(fontsize=5, loc="best")
            if ri == 0:
                ax.set_title(name, fontsize=6.5, pad=2)
            if fi == 0:
                ax.set_ylabel(f"{cell} {stimulus}\nΔF/F %", fontsize=6)
            if ri == n_r - 1:
                ax.set_xlabel("time (s)", fontsize=6)
            ax.tick_params(labelsize=5)

    fig.suptitle(title, y=0.998, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


def main() -> int:
    df = pd.read_csv(YANG_CSV)
    vdf = df[df.modality == "voltage"]
    cdf = df[df.modality == "calcium"]
    t_max_v = float(vdf.time_s.max()) + DT
    t_max_ca = float(cdf.time_s.max()) + DT
    t_grid_v = np.arange(0.0, t_max_v + 1e-9, DT)
    t_grid_ca = np.arange(0.0, t_max_ca + 1e-9, DT)
    n_f = len(FILTERS)

    plot_grid(
        vdf,
        t_grid_v,
        "full",
        None,
        OUT_FULL,
        f"Yang S5 V: filter on full digitized span (12×{n_f})",
    )
    plot_grid(
        vdf,
        t_grid_v,
        "tail",
        TAIL450_MS / 1000.0,
        OUT_TAIL450,
        f"Yang S5 V: filter on last {TAIL450_MS:g} ms only (12×{n_f})",
    )
    plot_grid(
        vdf,
        t_grid_ca,
        "tail_extrap",
        TAIL450_MS / 1000.0,
        OUT_TAIL450_EXTRAP,
        f"Yang S5 V: tail {TAIL450_MS:g} ms — Raw peak fit + smoothed filters (12×{n_f + 1})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
