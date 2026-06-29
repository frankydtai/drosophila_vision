"""Diagnostic: why does R's mirrored-Ih response decay so fast?
Replicates add_r_data's config (no training), loads a trained best_param, and
reports R1 vs L1 Ih reversal (E_Ih_eff), activation (Ih_ss) and kinetics (tau).
Analysis only -- reuses fc._run_conductance, does not re-implement the model."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
import FiveCol_MedSim_Pytorch as fc

RUN = sys.argv[1] if len(sys.argv) > 1 else 'FiveCol_Parameter/conductance/run_20260621_182857'
CT = np.load('Circuits/ctype.npy', allow_pickle=True)

# --- replicate add_r_data config that affects the schema/Ih (steps 2b + 3) ---
fc.MODEL_TYPE = 'conductance'
fc.IH_DIR_REVERSE_CELLS = sorted(set(fc.IH_DIR_REVERSE_CELLS) | set(range(8)))
fc.E_LEAK_DEPOL_CELLS = sorted(set(fc.E_LEAK_DEPOL_CELLS) | set(range(8)))
fc.E_leak = fc.build_e_leak(); fc.Ih_dir = fc.build_ih_dir()
groups = [[0, 1, 2, 3, 4, 5], 6, 7, 8, 9, 10, 11, 12]
schema = [dict(s) for s in fc.CONDUCTANCE_SCHEMA]
for s in schema:
    if s['name'] == 'Ih_gmax':
        s['cells'] = groups; s['zero'] = [groups.index(10), groups.index(11)]
fc.CONDUCTANCE_SCHEMA = schema

z = torch.tensor(np.load(os.path.join(RUN, 'best_param.npy')), dtype=torch.float64)
p = fc.assign_params(z, fc.CONDUCTANCE_SCHEMA)
mid = float(p['Ih_midv']); slope = float(p['Ih_slope']); tmid = float(p['tau_midv'])
center = 2 * fc.nofcells

def tau_ms(Vm):  # the u time-constant used in update_Vm
    return 1.5 / (np.exp(-0.1 * (Vm - tmid)) + np.exp(+0.1 * (Vm - tmid))) * 1000.0 + 100.0

print(f'globals: Ih_midv={mid:.1f}  Ih_slope={slope:.3f}  tau_midv={tmid:.1f}  E_Ih=+{fc.E_Ih}')
for name in ['R1', 'L1']:
    i = int(np.where(CT == name)[0][0]); dir_c = float(fc.Ih_dir[i + center])
    m, ref = fc._run_conductance(p, neuron_index=torch.tensor([i + center]), return_ref=True)
    m = m[:, 0].detach().numpy(); ref = float(ref)
    k = fc.Ca_tau / fc.deltat; mp = np.concatenate([[0.0], m[:-1]])
    Vm = (mp + (m - mp) * k) + ref                       # invert Ca lowpass -> Vm
    pk = int(np.argmax(np.abs(Vm - ref))); Vpk = Vm[pk]
    E_eff = mid + dir_c * (fc.E_Ih - mid)
    print(f'\n{name}: Ih_dir={dir_c:+.0f}  Vm_ref={ref:.1f}  Vm range=[{Vm.min():.1f},{Vm.max():.1f}]  peak Vm={Vpk:.1f}')
    print(f'   E_Ih_eff={E_eff:.1f} mV   driving force at peak (E_eff-Vpk)={E_eff - Vpk:.1f} mV')
    print(f'   tau(rest)={tau_ms(ref):.0f} ms   tau(peak)={tau_ms(Vpk):.0f} ms')
