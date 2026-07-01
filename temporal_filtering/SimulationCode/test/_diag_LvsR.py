"""Why does L decay but R not (adaptive)? Decompose the real update_state_adaptive
per step for R1 vs L1: drive X, gate, v_sustained, v_transient. Reuses the core
step function (instrumentation only, no model re-implementation)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc

RUN = 'FiveCol_Parameter/adaptive/run_20260621_185424'
CT = np.load('Circuits/ctype.npy', allow_pickle=True)
fc.MODEL_TYPE = 'adaptive'
groups = [[0, 1, 2, 3, 4, 5], 6, 7, 8, 9, 10, 11, 12]
sch = [dict(s) for s in fc.ADAPTIVE_SCHEMA]
for s in sch:
    if s['name'] in ('adapt_gain', 'tau_adapt'):
        s['cells'] = groups
fc.ADAPTIVE_SCHEMA = sch
z = torch.tensor(np.load(os.path.join(RUN, 'best_param.npy')), dtype=torch.float64)
p = fc.assign_params(z, fc.ADAPTIVE_SCHEMA); p['gate_pivot'] = fc.GATE_PIVOT

R1 = ml.center_unit_index(ml.type_index('R1'))
L1 = ml.center_unit_index(ml.type_index('L1'))
x_signal = fc.signal / ml.SIGNAL_BRIGHT
activity = p['bias'].clone(); v_sus = p['bias'].clone()
v_tra = torch.zeros_like(p['bias']); drive_lp = p['bias'].clone()

# replicate the per-step internals we want to log (mirrors update_state_adaptive)
def internals(act, dlp, x_t, x_d):
    syn = p['inp_gain'] * torch.mv(fc.M_signed, torch.relu(act) * p['out_gain'])
    X = p['bias'] + syn + x_t
    Xg = p['bias'] + syn + x_d
    gate = (Xg - p['gate_pivot']) * p['adapt_gain']
    return X, gate

log = {i: {'X': [], 'gate': [], 'sus': [], 'tra': []} for i in (R1, L1)}
for t in range(1, fc.maxtime):
    x_t = x_signal[t - 1]; x_d = x_signal[max(t - 1 - fc.gate_lag, 0)]
    X, gate = internals(activity, drive_lp, x_t, x_d)
    activity, v_sus, v_tra, drive_lp = fc.update_state_adaptive(
        activity, v_sus, v_tra, drive_lp, p, x_t, x_d)
    for i in (R1, L1):
        log[i]['X'].append(float(X[i])); log[i]['gate'].append(float(gate[i]))
        log[i]['sus'].append(float(v_sus[i])); log[i]['tra'].append(float(v_tra[i]))

for i, nm in ((R1, 'R1'), (L1, 'L1')):
    d = {k: np.array(v) for k, v in log[i].items()}
    on = slice(49, 199)               # stimulus on
    sus, tra = d['sus'][on], d['tra'][on]
    resp = sus + tra
    pk = resp[np.argmax(np.abs(resp - resp[0]))]
    print(f"\n{nm}: bias={float(p['bias'][i]):+.2f}  adapt_gain={float(p['adapt_gain'][i]):+.2f}")
    print(f"   X(rest->plateau)={d['X'][on][0]:+.2f} -> {d['X'][on][-30:].mean():+.2f}   "
          f"gate(on)~{d['gate'][on][5]:+.2f}")
    print(f"   v_sustained plateau={sus[-30:].mean():+.3f}   "
          f"v_transient peak={tra[np.argmax(np.abs(tra))]:+.3f} -> end={tra[-30:].mean():+.3f}")
    print(f"   response peak={pk:+.3f} -> plateau={resp[-30:].mean():+.3f}  "
          f"(transient/sustained = {abs(tra[np.argmax(np.abs(tra))]) / (abs(sus[-30:].mean()) + 1e-9):.2f})")
