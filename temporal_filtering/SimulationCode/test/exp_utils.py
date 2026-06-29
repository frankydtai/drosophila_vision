"""Bootstrap + tiny helpers shared by the analysis scripts in this folder.

No model forward / trainer / cost is defined here -- those live in the core
(fc.simulate_adaptive / fc.simulate_conductance / fc.model_cost / fc.train_staged,
and run.run_dir / run.write_param_table). This module only sets up the import
path and provides small experiment-only conveniences.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # experiments run on CPU
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)                    # .../SimulationCode
os.chdir(ROOT)                                   # fc loads Circuits/ relative to cwd
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import FiveCol_MedSim_Pytorch as fc

device = fc.device
STEPS = int(os.environ.get("EXP_STEPS", "150"))  # per-lr-stage steps (smoke-test override)
LRS = (0.05, 0.01, 0.002)


def best_scale_decomp(model):
    """Closed-form optimal global scale s* for a fixed waveform, and the losses
    at s=1 vs s=s*. (Diagnostic only -- not part of the model.)"""
    data = fc.data
    s_star = float(torch.sum(model * data[50:200]) / torch.sum(model * model))
    return s_star, float(fc.model_cost(model, data)), float(fc.model_cost(model, data, s_star))


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
