"""Bootstrap + tiny helpers shared by the analysis scripts in this folder."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import FiveCol_MedSim_Pytorch as fc

STEPS = int(os.environ.get("EXP_STEPS", "150"))
LRS = (0.05, 0.01, 0.002)


def best_scale_decomp(session, model):
    pack = session.primary_pack
    data = pack.data
    s_star = float(torch.sum(model * data) / torch.sum(model * model))
    return s_star, float(fc.model_cost(model, data, session)), float(
        fc.model_cost(model, data, session, scale=s_star)
    )


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
