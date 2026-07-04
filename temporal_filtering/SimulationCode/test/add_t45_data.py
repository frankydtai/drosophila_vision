"""Experiment: add T4a-d / T5a-d as fit cells that mirror L4 / L5."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

import FiveCol_MedSim_Pytorch as fc
import plot_trained as pt
from plot import tile as tile_plot
import run
from t4_t5_preference import READOUT_SUBTYPES

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--model_type', default='conductance', choices=['conductance', 'adaptive'])
ap.add_argument('--network', default=None, metavar='RUN',
                help='built_network run folder (network tile); omit for Borst 5-column')
ap.add_argument('--nofruns', type=int, default=1)
ap.add_argument('--nofsteps', type=int, default=100)
ap.add_argument('--lrs', default='0.1',
                help='comma-separated learning rates')
ap.add_argument(
    '--target',
    default='tile_bright',
    help="comma-separated targets: tile (=bright+dark), moving_bar (=bright+dark), or explicit names",
)
ap.add_argument('--per_type', action='store_true',
                help='train Ih (and adaptive lamina) params per cell type')
args = ap.parse_args()
MODEL = args.model_type
try:
    target_list = run.parse_target_list(args.target)
except ValueError as exc:
    ap.error(str(exc))
lrs = run.parse_comma_floats(args.lrs)
if not lrs:
    ap.error('--lrs must list at least one learning rate')

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
    t: dict(T45_MIRROR) for t in target_list if t in fc.TILE_TARGETS
}


def mirror_ref_cubes(dark=False):
    ref = pt.default_ref_cubes(dark=dark)
    for name in T4_NAMES:
        ref[name] = MIRROR_SIGN * ref['L4']
    for name in T5_NAMES:
        ref[name] = MIRROR_SIGN * ref['L5']
    return ref


tile_targets = [t for t in target_list if t in fc.TILE_TARGETS]
plot_ref_cubes = plot_ref_cubes_off = None
if 'tile_bright' in tile_targets and 'tile_dark' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)
    plot_ref_cubes_off = mirror_ref_cubes(dark=True)
elif 'tile_dark' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=True)
elif 'tile_bright' in tile_targets:
    plot_ref_cubes = mirror_ref_cubes(dark=False)

plot_mvd_groups = [
    np.array(T4_NAMES),
    np.array(T5_NAMES),
] + tile_plot.DEFAULT_MVD_GROUPS

fname, outdir, session = run.run_training(
    MODEL, args.nofruns, args.nofsteps, lrs,
    network=args.network,
    target_list=target_list,
    pack_overrides=PACK_OVERRIDES,
    per_type=args.per_type,
    plot_ref_cubes=plot_ref_cubes,
    plot_ref_cubes_off=plot_ref_cubes_off,
    plot_mvd_group_list=plot_mvd_groups,
)
for tname in tile_targets:
    print(f'{tname} cost cells:', int(session.pack_for(tname).readout_unit.shape[0]))
print('done ->', outdir)
