"""Experiment: add R1-8 photoreceptors as fit cells that mirror L1."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
import plot_trained as pt
from plot import tile as tile_plot
import run

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model_type', default='conductance', choices=['conductance', 'adaptive'])
ap.add_argument('--network', default=None, metavar='RUN',
                help='built_network run folder (network tile); omit for Borst 5-column')
ap.add_argument('--nofruns', type=int, default=1)
ap.add_argument('--nofsteps', type=int, default=100)
ap.add_argument('--lrs', type=float, nargs='+', default=[0.1])
args = ap.parse_args()
MODEL = args.model_type

CTYPE = np.load(os.path.join(ROOT, 'Circuits', 'ctype.npy'), allow_pickle=True)
R_NAMES = [str(CTYPE[i]) for i in range(ml.N_PHOTORECEPTORS)]
SHARED_R_NAMES = R_NAMES[:6]
INDEP_R_NAMES = R_NAMES[6:8]
MIRROR_FIT_CELL = 'L1'
MIRROR_SIGN = -1.0

PACK_OVERRIDES = {
    'tile': {
        'mirror_fit': {
            'mirror_types': R_NAMES,
            'mirror_fit': MIRROR_FIT_CELL,
            'mirror_sign': MIRROR_SIGN,
        },
    },
}


def group_lamina(schema, names, grp):
    out = [dict(s) for s in schema]
    for s in out:
        if s['name'] in names:
            s['cells'] = grp
    return out


r_shared = fc.resolve_type_indices(SHARED_R_NAMES, fc.borst_backend())
r_indep = fc.resolve_type_indices(INDEP_R_NAMES, fc.borst_backend())
lamina_default = fc.resolve_type_indices(
    [str(CTYPE[i]) for i in range(ml.LAMINA_SLICE.start, ml.LAMINA_SLICE.stop)],
    fc.borst_backend(),
)
groups = [r_shared] + [[i] for i in r_indep] + [[i] for i in lamina_default]
photoreceptor_types = fc.resolve_type_indices(R_NAMES, fc.borst_backend())

model_backend = None
if MODEL == 'conductance':
    depol = sorted(set(ml.LAMINA_DEPOL_TYPES) | set(photoreceptor_types))
    ih_rev = sorted(set(photoreceptor_types))
    model_backend = fc.borst_backend(depol_cells=depol, ih_reverse_cells=ih_rev)
    print('E_leak depol cells:', depol)
    print('Ih reversed cells:', ih_rev)
    schema = group_lamina(fc.default_schema('conductance', model_backend), ['Ih_gmax'], groups)
    for s in schema:
        if s['name'] == 'Ih_gmax':
            l3 = lamina_default[2]
            l4 = lamina_default[3]
            s['zero'] = [groups.index([l3]), groups.index([l4])]
    lamina_names = ['Ih_gmax']
else:
    mb = fc.borst_backend()
    schema = group_lamina(fc.default_schema('adaptive', mb), ['adapt_gain', 'tau_adapt'], groups)
    lamina_names = ['adapt_gain', 'tau_adapt']

for name in lamina_names:
    seg = next(s for s in schema if s['name'] == name)
    print('%s groups: %s  trainable values: %d' % (name, groups, fc.seg_count(seg)))

ref = pt.default_ref_cubes()
for name in R_NAMES:
    ref[name] = MIRROR_SIGN * ref[MIRROR_FIT_CELL]
tile_plot.REF_CUBES = ref
tile_plot.MVD_GROUPS = [np.array(R_NAMES)] + tile_plot.DEFAULT_MVD_GROUPS

fname, outdir, session = run.run_training(
    MODEL, args.nofruns, args.nofsteps, args.lrs,
    network=args.network,
    pack_overrides=PACK_OVERRIDES,
    model_backend=model_backend,
    schema=schema,
)
print('cost cells:', int(session.pack_for('tile').readout_unit.shape[0]))
print('done ->', outdir)
