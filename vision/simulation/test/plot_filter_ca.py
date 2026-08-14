"""Plot 25 ms-spot GT ImpR as ``v``, ``v_ca``, ``ca``, and Arenz ``gt_ca``.

For each of the 13 gt cells:

- ``v`` = ``GT_AMP * (RF_sign · ImpR_v − impr_offset)`` with ``filter=none`` ImpR
  (``RF_sign`` same as ``task.spot.gt.read_RecF_ImpR``)
- ``v_ca`` / ``ca`` from ``neuron.forward`` path::

      v_ca = relu(v − v_th_ca) · a_ca
      ca = filter_ca(v_ca)   # ca[0] = v_ca[0]

- ``gt_ca`` = ``GT_AMP * (ImpR_ca − impr_offset)`` with ``filter=ca`` (Arenz digitized)

Default ``v_th_ca`` / ``a_ca`` / ``tau_ca`` from ``NEURON_SCHEMA['optimizable']`` val;
``impr_offset`` default ``0``. Writes two PNGs: full overlay, and ``ca``
τ_ca sweep (100, 350, 500, 1000 ms) vs ``gt_ca``.

Usage (from ``simulation/``):

    ../.venv/bin/python test/plot_filter_ca.py
    ../.venv/bin/python test/plot_filter_ca.py --show
    ../.venv/bin/python test/plot_filter_ca.py --impr-offset 0.5 --a-ca 10
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

from figure.util import TRACE_LW, save_figure
from import_bootstrap import parse_comma_list
from network.construction import cell_order_rows
from neuron.filter_ca import filter_ca
from default_params import (
    GT_AMP,
    DELTA_MS,
    DELTA_MS_PRE,
    MS_PRE,
    MS_RESPONSE,
    MS_SPOT_CA,
    NEURON_SCHEMA,
)
from task.spot.gt import GT_CELLS, read_RecF_ImpR
from task.spot.input import spot_timing_t

DEFAULT_SAVE = os.path.join(HERE, "filter_ca.png")
DEFAULT_SAVE_TAU_SWEEP = os.path.join(HERE, "filter_ca_tau_sweep.png")
_OPTIMIZABLE = NEURON_SCHEMA['optimizable']
TAU_CA = float(_OPTIMIZABLE["tau_ca"]["val"])
A_CA = float(_OPTIMIZABLE["a_ca"]["val"])
V_TH_CA = float(_OPTIMIZABLE["v_th_ca"]["val"])
IMPR_OFFSET = 0
TAU_CA_SWEEP = (100.0, 350.0, 500.0, 1000.0)
# Same order as GT_CELLS / read_RecF_ImpR RF_sign.
RF_SIGN = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])


def v_ca_from_v(v: np.ndarray, *, v_th_ca: float, a_ca: float) -> np.ndarray:
    """``v_ca = relu(v − v_th_ca)·a_ca`` (same as ``neuron.forward.v_ca_from_v``)."""
    return np.maximum(np.asarray(v, dtype=np.float64) - float(v_th_ca), 0.0) * float(a_ca)


def ca_from_v_ca(v_ca: np.ndarray, *, delta_ms: float, tau_ca: float) -> np.ndarray:
    """Full-T ``filter_ca`` on ``v_ca``; ``ca[0] = v_ca[0]``."""
    v_ca = np.asarray(v_ca, dtype=np.float64)
    ca = np.empty_like(v_ca)
    ca[0] = v_ca[0]
    tau = float(tau_ca)
    dt = float(delta_ms)
    for t in range(1, v_ca.shape[0]):
        ca[t] = filter_ca(ca[t - 1], v_ca[t], delta_ms=dt, tau_ca=tau)
    return ca


def _plot(
    v_rt, v_ca_rt, ca_rt, gt_ca_rt, *,
    t_onset, delta_ms, ms_spot, tau_ca, a_ca, v_th_ca, impr_offset, save, show,
):
    active = [str(n) for n in GT_CELLS]
    groups = [np.array(row) for row in cell_order_rows(active)]
    nrows = len(groups)
    ncols = max(len(cell_group) for cell_group in groups)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows), squeeze=False,
    )
    t_s = (np.arange(v_rt.shape[1]) - t_onset) * delta_ms / 1000.0
    i_from_name = {str(n): i for i, n in enumerate(GT_CELLS)}

    for r, group in enumerate(groups):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(group):
                ax.axis("off")
                continue
            name = str(group[c])
            i = i_from_name[name]
            ax.plot(
                t_s, v_rt[i], color="C0", lw=TRACE_LW,
                label=f"v (GT_AMP·(RF_sign·ImpR−{impr_offset:g}))",
            )
            ax.plot(t_s, v_ca_rt[i], color="C3", lw=TRACE_LW, label="v_ca")
            ax.plot(t_s, ca_rt[i], color="C1", lw=TRACE_LW, label="ca")
            ax.plot(
                t_s, gt_ca_rt[i], color="C2", lw=TRACE_LW,
                label=f"gt_ca (GT_AMP·(ImpR−{impr_offset:g}))",
            )
            ys = np.concatenate([v_rt[i], ca_rt[i], gt_ca_rt[i]])
            lo = float(np.nanmin(ys))
            hi = float(np.nanmax(ys))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(name, fontsize=8)
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.axvline(0.0, color="0.85", lw=0.6, ls="--")
            ax.axvspan(0.0, ms_spot / 1000.0, color="0.92", zorder=0)
            if r == nrows - 1:
                ax.set_xlabel("t − onset (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel("amp", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(
        f"v / v_ca / ca / gt_ca  "
        f"(GT_AMP={GT_AMP:g}, impr_offset={impr_offset:g}, ms_spot={ms_spot:g}, "
        f"τ_ca={tau_ca:g} ms, a_ca={a_ca:g}, v_th_ca={v_th_ca:g}, Δt={delta_ms:g} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save)
    print(f"saved {save}")
    if show:
        plt.show()
    plt.close(fig)


def _plot_tau_sweep(
    ca_by_tau, gt_ca_rt, tau_list, *,
    t_onset, delta_ms, ms_spot, a_ca, v_th_ca, impr_offset, save, show,
):
    """``ca`` for each ``tau_ca`` in ``tau_list``, plus ``gt_ca``."""
    active = [str(n) for n in GT_CELLS]
    groups = [np.array(row) for row in cell_order_rows(active)]
    nrows = len(groups)
    ncols = max(len(cell_group) for cell_group in groups)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.2 * ncols, 2.0 * nrows), squeeze=False,
    )
    n_t = gt_ca_rt.shape[1]
    t_s = (np.arange(n_t) - t_onset) * delta_ms / 1000.0
    i_from_name = {str(n): i for i, n in enumerate(GT_CELLS)}
    tau_colors = [f"C{k}" for k in range(len(tau_list))]

    for r, group in enumerate(groups):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(group):
                ax.axis("off")
                continue
            name = str(group[c])
            i = i_from_name[name]
            ys = [gt_ca_rt[i]]
            for tau, color in zip(tau_list, tau_colors):
                tr = ca_by_tau[tau][i]
                ax.plot(
                    t_s, tr, color=color, lw=TRACE_LW,
                    label=f"ca τ={tau:g} ms",
                )
                ys.append(tr)
            ax.plot(
                t_s, gt_ca_rt[i], color="0.2", lw=TRACE_LW, ls="--",
                label=f"gt_ca (GT_AMP·(ImpR−{impr_offset:g}))",
            )
            stack = np.concatenate(ys)
            lo = float(np.nanmin(stack))
            hi = float(np.nanmax(stack))
            pad = 0.05 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(name, fontsize=8)
            ax.axhline(0.0, color="0.7", lw=0.5)
            ax.axvline(0.0, color="0.85", lw=0.6, ls="--")
            ax.axvspan(0.0, ms_spot / 1000.0, color="0.92", zorder=0)
            if r == nrows - 1:
                ax.set_xlabel("t − onset (s)", fontsize=8)
            if c == 0:
                ax.set_ylabel("amp", fontsize=8)
            ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    tau_txt = ",".join(f"{t:g}" for t in tau_list)
    fig.suptitle(
        f"ca τ_ca sweep vs gt_ca  "
        f"(τ_ca=[{tau_txt}] ms, GT_AMP={GT_AMP:g}, impr_offset={impr_offset:g}, "
        f"ms_spot={ms_spot:g}, a_ca={a_ca:g}, v_th_ca={v_th_ca:g}, Δt={delta_ms:g} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, save)
    print(f"saved {save}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save", default=DEFAULT_SAVE)
    ap.add_argument("--save-tau-sweep", default=DEFAULT_SAVE_TAU_SWEEP)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--ms-spot", type=float, default=MS_SPOT_CA)
    ap.add_argument("--tau-ca", type=float, default=TAU_CA)
    ap.add_argument(
        "--tau-ca-sweep",
        default=",".join(f"{t:g}" for t in TAU_CA_SWEEP),
        help="comma-separated τ_ca ms for second PNG "
             f"(default: {','.join(f'{t:g}' for t in TAU_CA_SWEEP)})",
    )
    ap.add_argument("--a-ca", type=float, default=A_CA)
    ap.add_argument("--v-th-ca", type=float, default=V_TH_CA)
    ap.add_argument("--impr-offset", type=float, default=IMPR_OFFSET)
    ap.add_argument("--delta-ms", type=float, default=DELTA_MS)
    ap.add_argument("--ms-pre", type=float, default=MS_PRE)
    ap.add_argument("--ms-response", type=float, default=MS_RESPONSE)
    args = ap.parse_args()

    delta_ms = float(args.delta_ms)
    ms_spot = float(args.ms_spot)
    tau_ca = float(args.tau_ca)
    a_ca = float(args.a_ca)
    v_th_ca = float(args.v_th_ca)
    impr_offset = float(args.impr_offset)
    tau_sweep = tuple(float(x) for x in parse_comma_list(args.tau_ca_sweep))
    if not tau_sweep:
        raise SystemExit("--tau-ca-sweep must not be empty")
    t_onset, n_t = spot_timing_t(
        ms_pre=float(args.ms_pre),
        ms_response=float(args.ms_response),
        delta_ms=delta_ms,
        delta_ms_pre=float(DELTA_MS_PRE),
    )
    _, impr_v = read_RecF_ImpR(
        t_onset=t_onset,
        n_t=n_t,
        ms_spot=ms_spot,
        delta_ms=delta_ms,
        filter="none",
    )
    _, impr_ca = read_RecF_ImpR(
        t_onset=t_onset,
        n_t=n_t,
        ms_spot=ms_spot,
        delta_ms=delta_ms,
        filter="ca",
    )
    v_rt = float(GT_AMP) * (RF_SIGN[:, None] * impr_v - impr_offset)
    gt_ca_rt = float(GT_AMP) * (impr_ca - impr_offset)
    v_ca_rt = np.stack(
        [v_ca_from_v(v_rt[i], v_th_ca=v_th_ca, a_ca=a_ca) for i in range(v_rt.shape[0])],
        axis=0,
    )
    ca_rt = np.stack(
        [
            ca_from_v_ca(v_ca_rt[i], delta_ms=delta_ms, tau_ca=tau_ca)
            for i in range(v_ca_rt.shape[0])
        ],
        axis=0,
    )
    _plot(
        v_rt, v_ca_rt, ca_rt, gt_ca_rt,
        t_onset=t_onset, delta_ms=delta_ms, ms_spot=ms_spot, tau_ca=tau_ca,
        a_ca=a_ca, v_th_ca=v_th_ca, impr_offset=impr_offset,
        save=args.save, show=args.show,
    )
    ca_by_tau = {
        tau: np.stack(
            [
                ca_from_v_ca(v_ca_rt[i], delta_ms=delta_ms, tau_ca=tau)
                for i in range(v_ca_rt.shape[0])
            ],
            axis=0,
        )
        for tau in tau_sweep
    }
    _plot_tau_sweep(
        ca_by_tau, gt_ca_rt, tau_sweep,
        t_onset=t_onset, delta_ms=delta_ms, ms_spot=ms_spot,
        a_ca=a_ca, v_th_ca=v_th_ca, impr_offset=impr_offset,
        save=args.save_tau_sweep, show=args.show,
    )


if __name__ == "__main__":
    main()
