"""Diagnostic: does R decay in the adaptive model?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); os.chdir(ROOT); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
from training_config import PARAMETER_DIR

RUN = sys.argv[1] if len(sys.argv) > 1 else str(PARAMETER_DIR / 'adaptive' / 'run_20260621_185424')
groups = [[0, 1, 2, 3, 4, 5], 6, 7, 8, 9, 10, 11, 12]
mb = fc.borst_backend()
schema = [dict(s) for s in fc.default_schema('adaptive', mb)]
for s in schema:
    if s['name'] in ('adapt_gain', 'tau_adapt'):
        s['cells'] = groups
session = fc.open_session(
    fc.make_train_opts(backend='borst', target_list=['tile']),
    'adaptive', schema=schema, model_backend=mb,
)

z = torch.tensor(np.load(os.path.join(RUN, 'best_param.npy')), dtype=torch.float64)
p0 = fc.assign_params_adaptive(z, list(session.schema), session.backend)
R_all = [ml.unit_index(c, r) for c in range(session.backend.n_cols) for r in range(ml.N_PHOTORECEPTORS)]
r1 = ml.center_unit_index(ml.type_index('R1'))

def trace(ag=None, ta=None):
    p = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in p0.items()}
    if ag is not None:
        p['adapt_gain'] = p['adapt_gain'].clone(); p['adapt_gain'][R_all] = ag
    if ta is not None:
        p['tau_adapt'] = p['tau_adapt'].clone(); p['tau_adapt'][R_all] = ta
    m = fc._run_adaptive(p, session, neuron_index=torch.tensor([r1]))[:, 0].detach().numpy()
    return m

base_ag = float(p0['adapt_gain'][r1]); base_ta = float(p0['tau_adapt'][r1])
print(f'trained R1: adapt_gain={base_ag:.3f}  tau_adapt={base_ta:.1f}\n')
print(f"{'adapt_gain':>10} {'tau_adapt':>9} {'peak':>8} {'final':>8} {'decay%':>7}")
for ag in [base_ag, -2.0, -5.0, -10.0, +2.0, +5.0]:
    for ta in [base_ta, 500.0]:
        m = trace(ag, ta)
        pk = m[np.argmax(np.abs(m))]; fin = m[-20:].mean()
        dec = 100.0 * (1 - fin / pk) if pk != 0 else 0.0
        print(f"{ag:>10.2f} {ta:>9.0f} {pk:>8.3f} {fin:>8.3f} {dec:>7.1f}")
