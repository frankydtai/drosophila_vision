"""L4 best-model trace vs the same trace with the Ca lowpass reverse-removed."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
from training_config import PARAMETER_DIR

session = fc.open_session(fc.make_train_opts(backend='borst', target_list=['spot_bright']), 'conductance')
schema = list(session.schema)

z138 = np.load(str(PARAMETER_DIR / 'with_Ih' / 'best_parameter.npy'))
n_outscale = fc.schema_nparams(schema) - z138.shape[0]
z = np.concatenate([z138, np.ones(n_outscale)])
z = torch.tensor(z, dtype=torch.float64, device=session.device)
print("padded params:", z138.shape[0], "->", z.shape[0], "(out_scale=1 x %d)" % n_outscale)

l4_cell = ml.type_index('L4')
idx = torch.tensor([ml.center_unit_index(l4_cell)], dtype=torch.long, device=session.device)

with torch.no_grad():
    p = fc.assign_params(z, schema, session.backend)
    stacked, vm_ref = fc._run_conductance(session, p, neuron_index=idx, return_ref=True)
m = stacked[:, 0]

k = fc.Ca_tau / fc.deltat
m_prev = torch.cat([torch.zeros(1, dtype=m.dtype, device=m.device), m[:-1]])
raw = m_prev + (m - m_prev) * k

cost = fc.calc_cost(z, session).item()
m_np, raw_np = m.cpu().numpy(), raw.cpu().numpy()
t = np.arange(fc.t_on, session.maxtime)

plt.figure(figsize=(9, 5))
plt.plot(t, m_np,   color="tab:red",  lw=2.5, label=r"L4 best model (post Ca-lpf, $\tau=5$ steps)")
plt.plot(t, raw_np, color="tab:blue", lw=1.8, ls="--", label=r"reverse-removed lpf (raw $V_m-V_{m,ref}$)")
plt.axhline(0, color="k", lw=0.6)
plt.axvline(50, color="gray", lw=0.6, ls=":")
plt.title("L4 conductance model: with vs without the Ca lowpass  (cost=%.2f%% data power)" % cost)
plt.xlabel("time step (x10 ms)")
plt.ylabel(r"response  ($V_m - V_{m,ref}$) [mV]")
plt.legend(loc="upper right", fontsize=9)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), "l4_model_lpf_removed.png")
plt.savefig(out, dpi=130)
print("saved ->", out)
