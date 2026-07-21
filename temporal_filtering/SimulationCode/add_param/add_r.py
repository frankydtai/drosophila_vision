"""Experiment: add R1-8 photoreceptors as fit cells that mirror L1."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import Medulla_Library as ml
import FiveCol_MedSim_Pytorch as fc
from plot.readout import borst_ref_cubes
import train
from param_defaults import DEFAULT_IH_GMAX_INDI_NAMES

ap = train.make_training_argparser(__doc__)
args = ap.parse_args()
try:
    train_kw = train.training_kwargs_from_args(args, script_stem='add_r')
except ValueError as exc:
    ap.error(str(exc))

target_list = train_kw['target_list']

_borst = bool(args.borst)
if _borst:
    R_NAMES = [str(ml.ctype[i]) for i in range(ml.N_PHOTORECEPTORS)]
else:
    R_NAMES = ['R1-6', 'R7', 'R8']
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
    t: dict(R_MIRROR) for t in target_list if t in fc.SPOT_TARGETS
}

IH_INDI_NAMES = list(DEFAULT_IH_GMAX_INDI_NAMES)

if train_kw['model'] == 'conductance':
    _ih_param_partitions = {
        name: {'indi': IH_INDI_NAMES, 'fixed': ['all']}
        for name in ('Ih_gmax', 'Ih_gmax_off')
    }
else:
    _ih_param_partitions = {
        name: {'indi': IH_INDI_NAMES, 'fixed': ['all']}
        for name in ('adapt_gain', 'tau_adapt')
    }

existing = train_kw.pop('param_partitions', None) or {}
param_partitions = {**existing, **_ih_param_partitions}


def mirror_ref_cubes(dark=False):
    ref = borst_ref_cubes(dark=dark)
    for name in R_NAMES:
        ref[name] = MIRROR_SIGN * ref['L1']
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

fname, outdir, session = train.run_training(
    **train_kw,
    borst=_borst,
    pack_overrides=PACK_OVERRIDES,
    param_partitions=param_partitions,
    plot_ref_cubes=plot_ref_cubes,
    plot_ref_cubes_2=plot_ref_cubes_2,
)
for tname in spot_targets:
    print(f'{tname} cost cells:', int(session.pack_for(tname).readout_unit.shape[0]))
print('done ->', outdir)
