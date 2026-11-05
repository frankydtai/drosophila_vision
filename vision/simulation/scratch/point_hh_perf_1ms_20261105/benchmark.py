"""Controlled local/A100 benchmark for single-step Point HH and hp_lp."""
from __future__ import annotations

import json
import hashlib
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

SIMULATION = Path(__file__).resolve().parents[2]
VISION = SIMULATION.parent
for path in (VISION, SIMULATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import import_bootstrap  # noqa: F401


def load_blob_module(name: str, blob: str):
    source = subprocess.check_output(
        ["git", "cat-file", "blob", blob],
        cwd=SIMULATION.parent.parent,
    )
    module = types.ModuleType(name)
    module.__file__ = f"git-blob:{blob}"
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    parent_name, _, child_name = name.rpartition(".")
    setattr(sys.modules[parent_name], child_name, module)
    return module


point_hh_blob = os.environ.get(
    "POINT_HH_BLOB", "6b7197ff4df687c318c2e7df065f040d7dfefc63"
)
forward_blob = os.environ.get(
    "FORWARD_BLOB", "c79a13df8427239d81776d4a126f991742957118"
)
if point_hh_blob or forward_blob:
    import neuron

    if point_hh_blob:
        point_hh_module = load_blob_module("neuron.point_hh", point_hh_blob)
    else:
        point_hh_module = sys.modules["neuron.point_hh"]
    if forward_blob:
        forward_module = load_blob_module("neuron.forward", forward_blob)
    else:
        forward_module = sys.modules["neuron.forward"]
    forward_module.MODEL_DRIVERS["point_hh"] = point_hh_module
    for exported in (
        "MODEL_DRIVERS",
        "MODELS",
        "ca_from_v_ca",
        "forward_ca",
        "forward_nodes",
        "forward_trace",
        "forward_v",
        "pack_t_onset",
        "step_delta_ms",
        "v_ca_from_v",
    ):
        setattr(neuron, exported, getattr(forward_module, exported))

import train
from neuron.forward import forward_v
from train.param import z_init_from_schema


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    source = (
        SIMULATION
        / "0_runs/point_hh/1105_220238-run-experiment-point_hh/data/train_opts.json"
    )
    opts = json.loads(source.read_text())
    for key in (
        "spread_sti_opts", "mbar_sti_opts", "sbar_sti_opts", "spot_sti_opts",
    ):
        if isinstance(opts.get(key), dict):
            opts[key]["delta_ms"] = 1.0
            opts[key]["delta_ms_pre"] = 1.0
    opts["point_hh"].pop("internal_delta_ms", None)
    opts.update(
        tasks=["spread"],
        contrasts=["bright"],
        network=None,
        fp=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sequential=not torch.cuda.is_available(),
        pre_grad=False,
    )

    results = []
    for model in ("hp_lp", "point_hh"):
        session = train.open_session(dict(opts), model)
        z = z_init_from_schema(session.schema, session.sim_dtype)
        params = train.params_from_z(z, session)
        pack = session.primary_pack
        stimulus = session.pack_i_sti(pack)

        with torch.no_grad():
            forward_v(session, params, stimulus, pack=pack)
            synchronize()
            times = []
            for _ in range(3):
                synchronize()
                start = time.perf_counter()
                trace = forward_v(session, params, stimulus, pack=pack)
                synchronize()
                times.append(time.perf_counter() - start)
        trace_cpu = trace.detach().cpu().contiguous()
        results.append(
            {
                "model": model,
                "seconds": times,
                "median_s": statistics.median(times),
                "trace_sha256": hashlib.sha256(
                    trace_cpu.numpy().tobytes()
                ).hexdigest(),
                "trace_min": float(trace_cpu.min()),
                "trace_max": float(trace_cpu.max()),
                "trace_sum": float(trace_cpu.double().sum()),
            }
        )

    payload = {
        "label": os.environ.get("BENCH_LABEL", "unlabelled"),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        ),
        "torch": torch.__version__,
        "point_hh_blob": point_hh_blob,
        "forward_blob": forward_blob,
        "results": results,
    }
    print("BENCHMARK_RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
