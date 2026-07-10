"""Experiment: add T4a-d / T5a-d as fit cells that mirror L4 / L5."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import FiveCol_MedSim_Pytorch as fc
import plot_trained as pt
from plot import spot as spot_plot
import train
from t4_t5_preference import READOUT_SUBTYPES

ap = train.make_training_argparser(__doc__)
args = ap.parse_args()
try:
    train_kw = train.training_kwargs_from_args(args, script_stem='add_t45')
except ValueError as exc:
    ap.error(str(exc))

target_list = train_kw['target_list']

T4_NAMES = [n for n in READOUT_SUBTYPES if n.startswith('T4')]
T5_NAMES = [n for n in READOUT_SUBTYPES if n.startswith('T5')]
MIRROR_SIGN = -1.0

T45_MIRROR = {
    'mirror_fits': [
        {
            'mirror_types': T4_NAMES,
            'mirror_fit': 'L4',
            'mirror_sign': MIRROR_SIGN,
        },
        {
            'mirror_types': T5_NAMES,
            'mirror_fit': 'L5',
            'mirror_sign': MIRROR_SIGN,
        },
    ],
}

PACK_OVERRIDES = {
    t: dict(T45_MIRROR) for t in target_list if t in fc.SPOT_TARGETS
}


def mirror_ref_cubes(dark=False):
    ref = pt.borst_ref_cubes(dark=dark)
    for name in T4_NAMES:
        ref[name] = MIRROR_SIGN * ref['L4']
    for name in T5_NAMES:
        ref[name] = MIRROR_SIGN * ref['L5']
    return ref


spot_targets = [t for t in target_list if t in fc.SPOT_TARGETS]
plot_ref_cubes = plot_ref_cubes_2 = None
if 'spot_bright' in spot_targets and 'spot_dark' in spot_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)
    plot_ref_cubes_2 = mirror_ref_cubes(dark=True)
elif 'spot_dark' in spot_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=True)
elif 'spot_bright' in spot_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)

plot_mvd_groups = [
    np.array(T4_NAMES),
    np.array(T5_NAMES),
] + spot_plot.DEFAULT_MVD_GROUPS

fname, outdir, session = train.run_training(
    **train_kw,
    pack_overrides=PACK_OVERRIDES,
    plot_ref_cubes=plot_ref_cubes,
    plot_ref_cubes_2=plot_ref_cubes_2,
    plot_mvd_group_list=plot_mvd_groups,
)
for tname in spot_targets:
    print(f'{tname} cost cells:', int(session.pack_for(tname).readout_unit.shape[0]))
print('done ->', outdir)
