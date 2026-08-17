"""Plot the spot model-gt figure for the selected borst run.

Usage (from ``SimulationCode/``):

    ../.venv/bin/python scratch/plot_with_Ih_spot.py
    ../.venv/bin/python scratch/plot_with_Ih_spot.py --show
    ../.venv/bin/python scratch/plot_with_Ih_spot.py --run-path borst/27849055-add_spot_r
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import import_bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

import FiveCol_MedSim_Pytorch as fc
from plot import spot as spot_plot
import plot_trained

DEFAULT_RUN_PATH = "borst/27849055-add_spot_r"
DEFAULT_SAVE = os.path.join(HERE, "model_spot.png")
SECOND_BRIGHT_MS = 50.0


def _cost_extent_hex_tag(cost_extent, n_cost_hexes) -> str:
    """Backfill missing tag helper without modifying core code."""
    extent_tag = "all hexes" if cost_extent is None else f"extent={int(cost_extent)}"
    if isinstance(n_cost_hexes, dict):
        cols = ", ".join(
            f"b{int(batch)}={int(count)}"
            for batch, count in sorted(n_cost_hexes.items())
        )
        return f"cost hexes per batch [{cols}], {extent_tag}"
    return f"{int(n_cost_hexes)} cost hexes, {extent_tag}"


def _network_spot_trace_readout(
    session,
    z,
    *,
    at_xs=None,
    at_ys=None,
    save_trace_csv_dir=None,
    trace_kind="model",
    show_pre=True,
):
    """Backfill missing plot.spot entrypoint without modifying core code."""
    t0 = time.perf_counter()
    at_x = at_xs[0] if at_xs else None
    at_y = at_ys[0] if at_ys else None
    rows = spot_plot._spot_forward_rows(
        session,
        z,
        trace_kind=trace_kind,
        save_trace_csv_dir=save_trace_csv_dir,
        at_x=at_x,
        at_y=at_y,
        show_pre=show_pre,
    )
    cells, group_rows, mt = spot_plot._spot_rt_from_rows(rows, session)
    slice_overlay = slice_labels = None
    if at_xs is not None or at_ys is not None:
        slice_overlay, slice_labels = spot_plot._spot_slice_overlay(
            rows,
            rows["batches"],
            at_xs,
            at_ys,
        )
    return spot_plot.SpotTraceReadout(
        cells=cells,
        group_rows=group_rows,
        session=session,
        slice_overlay=slice_overlay,
        slice_labels=slice_labels,
        slice_xs=at_xs,
        slice_ys=at_ys,
        n_t=mt,
        prep_s=time.perf_counter() - t0,
        v_th_by_cell=spot_plot.v_th_by_cell(z, session),
    )


def _annotate_baseline_zero_no_dashed(ax, _baseline):
    """Force baseline label to 0 and remove y=0 dashed line."""
    ylo, yhi = ax.get_ylim()
    ax.set_yticks([ylo, 0.0, yhi])
    ax.set_yticklabels([f'{ylo:+.0f}', '0.0', f'{yhi:+.0f}'], fontsize=6)
    # Intentionally do NOT draw ax.axhline here.


def _plot_cell_pair_second_bright_dashed(*args, **kwargs):
    """Render the second bright model as a dashed overlay."""
    kwargs["linestyle_2"] = "--"
    kwargs["label_2_model"] = f"{int(SECOND_BRIGHT_MS)} ms bright model"
    return _ORIG_PLOT_CELL_PAIR(*args, **kwargs)


def _plot_cell_pair_slices_second_bright_dashed(*args, **kwargs):
    """Render the second bright slice overlays as dashed."""
    kwargs["linestyle_2"] = "--"
    kwargs["label_2_scope"] = f"{int(SECOND_BRIGHT_MS)} ms bright scope"
    return _ORIG_PLOT_CELL_PAIR_SLICES(*args, **kwargs)


_ORIG_PLOT_CELL_PAIR = spot_plot.plot_cell_pair
_ORIG_PLOT_CELL_PAIR_SLICES = spot_plot.plot_cell_pair_slices


def _session_with_bright_ms_pulse(session, ms_pulse):
    """Return a copy of *session* whose bright sti drops to baseline after *ms_pulse*."""
    pack = session.primary_pack
    opts = dict((session.train_opts or {}).get("spot_bright_sti_opts") or {})
    from neuron.borst import t_from_ms
    delta_ms = float(opts["delta_ms"])
    t_on = t_from_ms(float(opts["ms_pre"]), delta_ms=delta_ms)
    pulse_t = max(1, int(round(float(ms_pulse) / delta_ms)))
    signal = pack.signal.clone()
    baseline = signal[:, :1, :].clone()
    t_off = min(int(signal.shape[1]), t_on + pulse_t)
    signal[:, t_off:, :] = baseline
    pack_short = replace(pack, signal=signal)
    targets = dict(session.targets)
    targets[pack_short.name] = pack_short
    return replace(session, targets=targets, cost_subpacks={}, fused_conductance=())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-path",
        default=DEFAULT_RUN_PATH,
        help="run folder under 0_runs/ or absolute path",
    )
    p.add_argument(
        "--save",
        default=DEFAULT_SAVE,
        help="output PNG path",
    )
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fc._cost_extent_hex_tag = _cost_extent_hex_tag
    spot_plot.network_spot_trace_readout = _network_spot_trace_readout
    spot_plot.annotate_baseline = _annotate_baseline_zero_no_dashed
    spot_plot.plot_cell_pair = _plot_cell_pair_second_bright_dashed
    spot_plot.plot_cell_pair_slices = _plot_cell_pair_slices_second_bright_dashed
    run_dir = plot_trained.resolve_run_dir(args.run_path)
    save_path = os.path.abspath(args.save)
    outdir = os.path.dirname(save_path)
    if not outdir:
        raise SystemExit("--save must include a filename")
    os.makedirs(outdir, exist_ok=True)

    session, z_best, _best_i, best_cost = plot_trained.load_best(run_dir, verbose=True)
    bright_session = plot_trained.session_for_target(session, "spot_bright")
    bright_50ms_session = _session_with_bright_ms_pulse(bright_session, SECOND_BRIGHT_MS)
    readout = spot_plot.network_spot_trace_readout(bright_session, z_best)
    readout_50ms = spot_plot.network_spot_trace_readout(bright_50ms_session, z_best)
    title = (
        f"spot_bright model-gt (trained, cost {best_cost:.2f}% of gt power)"
        f"{plot_trained._network_spot_tag(bright_session, 'spot_bright')}"
    )
    spot_plot.plot_network_spot_gt(
        save_path,
        readouts={'bright': readout, 'bright_50ms': readout_50ms},
        title=title,
        ref_rts=spot_plot.spot_ref_rts(bright_session, "spot_bright", dark=False),
        ref_rts_2={},
    )
    print(f"saved {save_path}")
    if args.show:
        img = plt.imread(save_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
