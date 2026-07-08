"""Diagnostic: why does R's mirrored-Ih response decay so fast?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
from training_config import PARAMETER_DIR

RUN = sys.argv[1] if len(sys.argv) > 1 else str(PARAMETER_DIR / 'conductance' / 'run_20260621_182857')

depol = sorted(set(fc.E_LEAK_DEPOL_CELLS) | set(range(8)))
ih_rev = sorted(set(fc.IH_DIR_REVERSE_CELLS) | set(range(8)))
mb = fc.borst_backend(depol_cells=depol, ih_reverse_cells=ih_rev)
groups = [[0, 1, 2, 3, 4, 5], 6, 7, 8, 9, 10, 11, 12]
schema = [dict(s) for s in fc.default_schema('conductance', mb)]
for s in schema:
    if s['name'] == 'Ih_gmax':
        s['ih_group'] = groups; s['zero'] = [groups.index(10), groups.index(11)]
session = fc.open_session(
    fc.make_train_opts(backend='borst', target_list=['spot_bright']),
    'conductance', schema=schema, model_backend=mb,
)

z = torch.tensor(np.load(os.path.join(RUN, 'best_param.npy')), dtype=torch.float64)
schema = list(session.schema)
p = fc.assign_params(z, schema, session.backend)
mid = float(p['Ih_midv']); slope = float(p['Ih_slope']); tmid = float(p['tau_midv'])

def tau_ms(Vm):
    return 1.5 / (np.exp(-0.1 * (Vm - tmid)) + np.exp(+0.1 * (Vm - tmid))) * 1000.0 + 100.0

print(f'globals: Ih_midv={mid:.1f}  Ih_slope={slope:.3f}  tau_midv={tmid:.1f}  E_Ih=+{fc.E_Ih}')
for name in ['R1', 'L1']:
    unit = ml.center_unit_index(ml.type_index(name))
    dir_c = float(session.backend.ih_dir[unit])
    m, ref = fc._run_conductance(session, p, neuron_index=torch.tensor([unit]), return_ref=True)
    m = m[:, 0].detach().numpy(); ref = float(ref)
    k = fc.Ca_tau / fc.deltat; mp = np.concatenate([[0.0], m[:-1]])
    Vm = (mp + (m - mp) * k) + ref
    pk = int(np.argmax(np.abs(Vm - ref))); Vpk = Vm[pk]
    E_eff = mid + dir_c * (fc.E_Ih - mid)
    print(f'\n{name}: Ih_dir={dir_c:+.0f}  Vm_ref={ref:.1f}  Vm range=[{Vm.min():.1f},{Vm.max():.1f}]  peak Vm={Vpk:.1f}')
    print(f'   E_Ih_eff={E_eff:.1f} mV   driving force at peak (E_eff-Vpk)={E_eff - Vpk:.1f} mV')
    print(f'   tau(rest)={tau_ms(ref):.0f} ms   tau(peak)={tau_ms(Vpk):.0f} ms')
