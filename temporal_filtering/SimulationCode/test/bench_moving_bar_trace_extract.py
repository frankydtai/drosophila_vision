#!/usr/bin/env python3
"""Benchmark moving-bar trace extraction: legacy triple-loop vs vectorized path."""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

import network_bootstrap  # noqa: F401
import FiveCol_MedSim_Pytorch as fc
from connectome_io import DEFAULT_NETWORK_RUN, resolve_network_json
from network.moving_bar_target import _TRACE_CACHE
from network.stimulus import center_photo_column, photo_columns
from network.tiling import unit_type_names
from plot.moving_bar import (
    _aggregate_moving_bar_traces,
    _extract_moving_bar_windows,
    _moving_bar_center_only,
    _moving_bar_t0_grid,
    _network_type_ids,
)
from training_config import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from visual_stimulus.moving_bar_stimulus import gruntman_moving_bar_specs, column_bar_center_step


def _legacy_unit_window_trace(model_bt, u, t0):
    t_rel = t0 - fc.t_on + np.arange(COST_WINDOW_STEPS)
    t_max = model_bt.shape[0] - 1
    pre = t_rel < 0
    t_safe = np.clip(t_rel, 0, t_max)
    y = model_bt[t_safe, u].astype(np.float64, copy=True)
    y[pre] = 0.0
    return y


def _extract_legacy(model_full, C, cols, specs, types, center_only, center_col, t0_map):
    names_by_unit = unit_type_names(C)
    col_uv = {(int(c.u), int(c.v)) for c in cols}
    model_mean, model_sem = {}, {}
    for tname in types:
        for bi, spec in enumerate(specs):
            traces = []
            for u in range(C.n_units):
                if names_by_unit[u] != tname:
                    continue
                uv = (int(C.u[u]), int(C.v[u]))
                if center_only:
                    if uv != (int(center_col.u), int(center_col.v)):
                        continue
                elif uv not in col_uv:
                    continue
                t0 = t0_map.get((bi, uv[0], uv[1]))
                if t0 is None:
                    continue
                traces.append(_legacy_unit_window_trace(model_full[bi], u, t0))
            if not traces:
                continue
            arr = np.stack(traces, axis=0)
            key = (tname, spec.name)
            model_mean[key] = arr.mean(axis=0)
            if center_only or arr.shape[0] == 1:
                model_sem[key] = np.zeros(COST_WINDOW_STEPS)
            else:
                model_sem[key] = arr.std(axis=0) / np.sqrt(arr.shape[0])
    return model_mean, model_sem


def _extract_vectorized(model_full, C, cols, specs, types, center_only, t0_map):
    type_ids = _network_type_ids(C)
    spec_names = [s.name for s in specs]
    t0_bn = _moving_bar_t0_grid(C, cols, len(specs), t0_map)
    windows = _extract_moving_bar_windows(model_full, t0_bn)
    return _aggregate_moving_bar_traces(
        windows, t0_bn, type_ids, types, spec_names, center_only,
    )


def _bench(fn, n_repeat=5):
    times = []
    out = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, times


def main():
    _TRACE_CACHE.clear()
    network_json = resolve_network_json(DEFAULT_NETWORK_RUN)
    session = fc.open_session(
        fc.make_train_opts(
            backend="network", target_list=["moving_bar_bright"],
            network=fc.load_network_backend(network_json, dev="cpu").network,
            network_json=network_json,
            multi_column=False, sequential=True, dev="cpu",
        ),
        "conductance",
    )
    C = session.backend.network
    specs = gruntman_moving_bar_specs(contrasts=("bright",))
    center_only = _moving_bar_center_only(session)
    center_col = center_photo_column(C)
    cols = [center_col] if center_only else photo_columns(C)
    types = list(C.type_names)
    field_deg = C.meta.get("field_deg")
    if field_deg is None:
        from network.stimulus import build_moving_bar_signals
        field_deg = build_moving_bar_signals(
            C, t_on=fc.t_on, deltat_ms=fc.deltat, device="cpu",
        ).info["field_deg"]

    t0_map = {}
    for bi, spec in enumerate(specs):
        for c in cols:
            t_center = column_bar_center_step(
                c.x, c.y, spec, field_deg, t_on=fc.t_on, deltat_ms=fc.deltat,
            )
            t0_map[(bi, int(c.u), int(c.v))] = int(t_center - COST_HALF_WINDOW_STEPS)

    z = fc.guess_initial_params(session)
    p = fc.assign_params(z, list(session.schema), session.backend)
    t_fwd0 = time.perf_counter()
    pack = session.pack_for("moving_bar_bright")
    model_full = fc._run_conductance_full(session, p, pack.signal).cpu().numpy()
    t_forward = time.perf_counter() - t_fwd0

    legacy_out, legacy_times = _bench(
        lambda: _extract_legacy(
            model_full, C, cols, specs, types, center_only, center_col, t0_map,
        ),
    )
    new_out, new_times = _bench(
        lambda: _extract_vectorized(
            model_full, C, cols, specs, types, center_only, t0_map,
        ),
    )

    for k in legacy_out[0]:
        assert np.allclose(legacy_out[0][k], new_out[0][k], atol=1e-12), k
        assert np.allclose(legacy_out[1][k], new_out[1][k], atol=1e-12), k

    print(f"forward: {t_forward:.2f}s")
    print(f"legacy extract: mean={np.mean(legacy_times):.3f}s  panels={len(legacy_out[0])}")
    print(f"vectorized:     mean={np.mean(new_times):.3f}s  speedup={np.mean(legacy_times)/np.mean(new_times):.1f}x")
    print("ok")


if __name__ == "__main__":
    main()
