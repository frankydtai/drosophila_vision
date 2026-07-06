"""Experiment: add R1-8 photoreceptors as fit cells that mirror L1."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import FiveCol_MedSim_Pytorch as fc
import Medulla_Library as ml
import plot_trained as pt
from plot import tile as tile_plot
import train

ap = train.make_training_argparser(__doc__)
args = ap.parse_args()
try:
    train_kw = train.training_kwargs_from_args(args, script_stem='add_r')
except ValueError as exc:
    ap.error(str(exc))

target_list = train_kw['target_list']
MODEL = train_kw['model_type']

R_NAMES = [str(ml.ctype[i]) for i in range(ml.N_PHOTORECEPTORS)]
SHARED_R_NAMES = R_NAMES[:6]
INDEP_R_NAMES = R_NAMES[6:8]
MIRROR_SIGN = -1.0

R_MIRROR = {
    'mirror_fits': [
        {
            'mirror_types': R_NAMES,
            'mirror_fit': 'L1',
            'mirror_sign': MIRROR_SIGN,
        },
    ],
}

PACK_OVERRIDES = {
    t: dict(R_MIRROR) for t in target_list if t in fc.TILE_TARGETS
}


def mirror_ref_cubes(dark=False):
    ref = pt.borst_ref_cubes(dark=dark)
    for name in R_NAMES:
        ref[name] = MIRROR_SIGN * ref['L1']
    return ref


def group_lamina(schema, names, grp):
    out = [dict(s) for s in schema]
    for s in out:
        if s['name'] in names:
            s['cells'] = grp
    return out


mb = fc.borst_backend()
r_shared = fc.resolve_type_indices(SHARED_R_NAMES, mb)
r_indep = fc.resolve_type_indices(INDEP_R_NAMES, mb)
lamina_types = fc.resolve_type_indices(
    [str(ml.ctype[i]) for i in range(ml.LAMINA_SLICE.start, ml.LAMINA_SLICE.stop)],
    mb,
)
groups = [r_shared] + [[i] for i in r_indep] + [[i] for i in lamina_types]

if MODEL == 'conductance':
    lamina_names = ['Ih_gmax', 'Ih_gmax_off']
    schema = group_lamina(fc.default_schema('conductance', mb), lamina_names, groups)
    ih_zero = fc.lamina_zero_indices(groups, fc.IH_GMAX_ZERO_TYPES, ml.ctype)
    for s in schema:
        if s['name'] in lamina_names:
            s['zero'] = ih_zero
else:
    lamina_names = ['adapt_gain', 'tau_adapt']
    schema = group_lamina(fc.default_schema('adaptive', mb), lamina_names, groups)

for name in lamina_names:
    seg = next(s for s in schema if s['name'] == name)
    print('%s groups: %s  trainable values: %d' % (name, groups, fc.seg_count(seg)))

tile_targets = [t for t in target_list if t in fc.TILE_TARGETS]
plot_ref_cubes = plot_ref_cubes_off = None
if 'tile_bright' in tile_targets and 'tile_dark' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)
    plot_ref_cubes_off = mirror_ref_cubes(dark=True)
elif 'tile_dark' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=True)
elif 'tile_bright' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)

plot_mvd_groups = [np.array(R_NAMES)] + tile_plot.DEFAULT_MVD_GROUPS

fname, outdir, session = train.run_training(
    **train_kw,
    pack_overrides=PACK_OVERRIDES,
    schema=schema,
    plot_ref_cubes=plot_ref_cubes,
    plot_ref_cubes_off=plot_ref_cubes_off,
    plot_mvd_group_list=plot_mvd_groups,
)
for tname in tile_targets:
    print(f'{tname} cost cells:', int(session.pack_for(tname).readout_unit.shape[0]))
print('done ->', outdir)
