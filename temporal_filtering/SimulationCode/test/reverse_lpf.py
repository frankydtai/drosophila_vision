"""L4 best-model trace vs the same trace with the Ca lowpass reverse-removed.

Reuses the core forward (fc._run_conductance) for the post-lpf trace, then
inverts the explicit-Euler Ca lowpass
    m_t = m_{t-1} + (deltat/Ca_tau) * (raw_t - m_{t-1})
=>  raw_t = m_{t-1} + (m_t - m_{t-1}) * Ca_tau/deltat      (m_{-1}=0)
to recover the pre-lpf membrane trace. No model forward is re-implemented."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import FiveCol_MedSim_Pytorch as fc

fc.MODEL_TYPE = 'conductance'

# --- paper best params (138) padded with out_scale=1 -> current 203 schema ---
z138 = np.load('FiveCol_Parameter/with_Ih/best_parameter.npy')
n_outscale = fc.schema_nparams(fc.CONDUCTANCE_SCHEMA) - z138.shape[0]
z = np.concatenate([z138, np.ones(n_outscale)])
z = torch.tensor(z, dtype=torch.float64, device=fc.device)
print("padded params:", z138.shape[0], "->", z.shape[0], "(out_scale=1 x %d)" % n_outscale)

# --- L4 center-column neuron index in the 325-cell state ---
CTYPE = np.load('Circuits/ctype.npy', allow_pickle=True)
l4_cell = int(np.where(CTYPE == 'L4')[0][0])
CENTER_OFFSET = 2 * fc.nofcells                       # center of the 5 columns
idx = torch.tensor([l4_cell + CENTER_OFFSET], dtype=torch.long, device=fc.device)

# --- post-lpf trace straight from the core forward ---
with torch.no_grad():
    p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
    stacked, vm_ref = fc._run_conductance(p, neuron_index=idx, return_ref=True)
m = stacked[:, 0]                                     # (150,) model output, t=50..199

# --- reverse-remove the Ca lowpass (exact inverse of the Euler recurrence) ---
k = fc.Ca_tau / fc.deltat                             # = 5 steps
m_prev = torch.cat([torch.zeros(1, dtype=m.dtype, device=m.device), m[:-1]])
raw = m_prev + (m - m_prev) * k                       # pre-lpf membrane trace

cost = fc.calc_cost(z, fc.data).item()
m_np, raw_np = m.cpu().numpy(), raw.cpu().numpy()
t = np.arange(50, 200)

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
