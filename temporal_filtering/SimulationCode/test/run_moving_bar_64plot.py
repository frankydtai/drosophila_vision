"""Plot all 64 moving-bar model traces (8 subtypes × 8 stimuli), no training."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import network_bootstrap  # noqa: F401
import FiveCol_MedSim_Pytorch as fc
from connectome_io import DEFAULT_NETWORK_RUN, resolve_network_json
from network.moving_bar_target import _TRACE_CACHE
from network.construction import unit_type_names
from plot.moving_bar import _moving_bar_mean_traces
from plot_trained import plot_moving_bar_data, run_dir
from t4_t5_preference import READOUT_SUBTYPES, active_stimuli_for_subtype, normalize_side
from training_config import COST_HALF_WINDOW_STEPS, COST_WINDOW_STEPS
from visual_stimulus.moving_bar_stimulus import gruntman_moving_bar_specs


def _panel_keys(side: str):
    keys = []
    for st in READOUT_SUBTYPES:
        for d, c, w in active_stimuli_for_subtype(side, st):
            keys.append((st, f"{d}_{c}_{w}"))
    return keys


def _direct_mean_traces(session, z, target):
    """Independent panel extraction for one contrast pack."""
    contrast = "bright" if "bright" in target else "dark"
    specs = gruntman_moving_bar_specs(contrasts=(contrast,))
    spec_names = [s.name for s in specs]
    pack = session.pack_for(target)
    p = fc.assign_params(z, list(session.schema), session.backend)
    model_full = fc._run_conductance_full(session, p, pack.signal)
    model_sel = fc._readout_model_traces_pack(model_full, pack).cpu().numpy()
    u_idx = pack.readout_unit
    C = session.backend.network
    subtypes = unit_type_names(C)[u_idx.cpu().numpy()]
    batches = pack.readout_batch.cpu().numpy()
    groups: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(subtypes, [spec_names[b] for b in batches])):
        groups.setdefault(key, []).append(i)
    return {k: model_sel[idx].mean(axis=0) for k, idx in groups.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default=DEFAULT_NETWORK_RUN,
                    help="built_network run folder name")
    ap.add_argument("--outdir", default=None, help="output dir (default: fresh run_<id> under conductance/)")
    args = ap.parse_args()

    _TRACE_CACHE.clear()
    network_json = resolve_network_json(args.network)
    session = fc.open_session(
        fc.make_train_opts(
            backend="network",
            target_list=["moving_bar_bright", "moving_bar_dark"],
            network=fc.load_network_backend(network_json, dev="cpu").network,
            network_json=network_json,
            multi_column=False, sequential=True, dev="cpu",
        ),
        "conductance",
    )
    s_bright = fc.open_session({**session.train_opts, "target_list": ["moving_bar_bright"],
                                "backend": "network", "network": session.backend.network,
                                "packs": None}, "conductance",
                               model_backend=session.backend)
    s_dark = fc.open_session({**session.train_opts, "target_list": ["moving_bar_dark"],
                              "backend": "network", "network": session.backend.network,
                              "packs": None}, "conductance",
                             model_backend=session.backend)
    side = normalize_side(session.backend.network.meta.get("side", "right"))
    panels = _panel_keys(side)
    assert len(panels) == 64, f"expected 64 panels, got {len(panels)}"

    z = fc.guess_initial_params(session)
    row_specs_b, model_mean_b, model_sem_b, data_mean_b = _moving_bar_mean_traces(
        s_bright, z, "moving_bar_bright",
    )
    row_specs_d, model_mean_d, model_sem_d, data_mean_d = _moving_bar_mean_traces(
        s_dark, z, "moving_bar_dark",
    )
    model_mean = {**model_mean_b, **model_mean_d}
    data_mean = {**data_mean_b, **data_mean_d}
    direct = {
        **_direct_mean_traces(s_bright, z, "moving_bar_bright"),
        **_direct_mean_traces(s_dark, z, "moving_bar_dark"),
    }

    missing = [k for k in panels if k not in model_mean]
    assert not missing, f"missing model panels: {missing[:5]}..."

    for key in panels:
        m = model_mean[key]
        d = data_mean[key]
        assert m.shape == (COST_WINDOW_STEPS,)
        assert d.shape == (COST_WINDOW_STEPS,)
        assert np.allclose(m, direct[key], atol=1e-12), key
        assert np.isfinite(m).all() and np.isfinite(d).all()

    for st in READOUT_SUBTYPES:
        bright_specs = [f"{d}_{c}_{w}" for d, c, w in active_stimuli_for_subtype(side, st) if c == "bright"]
        dark_specs = [f"{d}_{c}_{w}" for d, c, w in active_stimuli_for_subtype(side, st) if c == "dark"]
        assert row_specs_b[st] == bright_specs
        assert row_specs_d[st] == dark_specs

    outdir = args.outdir or run_dir("conductance", parent=os.path.join("FiveCol_Parameter", "moving_bar_64plot"))
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, "model_data_bar.png")
    plot_moving_bar_data(s_bright, z, png, session_2=s_dark,
                         title="Moving-bar model-data (initial params)")

    peaks = []
    for key in panels:
        m = model_mean[key]
        d = data_mean[key]
        im = int(np.argmax(np.abs(m)))
        id_ = int(np.argmax(np.abs(d)))
        peaks.append((key, (im - COST_HALF_WINDOW_STEPS) * 0.01, (id_ - COST_HALF_WINDOW_STEPS) * 0.01))

    pack_b = s_bright.pack_for("moving_bar_bright")
    pack_d = s_dark.pack_for("moving_bar_dark")
    print(f"network run: {args.network}")
    print(f"panels: {len(panels)}  n_cost: {pack_b.data.shape[0] + pack_d.data.shape[0]}  outdir: {outdir}")
    print("sample peaks (model_s, data_s) rel. bar center:")
    for key, pm, pd in peaks[:4]:
        print(f"  {key[0]:4s} {key[1]:22s}  model={pm:+.2f}  data={pd:+.2f}")
    print(f"... ({len(peaks)} total)")
    assert os.path.isfile(png), png
    print("ok", png)


if __name__ == "__main__":
    main()
