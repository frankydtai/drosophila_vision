"""Experiment: add spot fit cells that mirror lamina (r→L1, t45→L4/L5)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import import_bootstrap  # noqa: F401
import training.driver as train
from connectome_io import parse_comma_list
from task.moving_bar.data import READOUT_SUBTYPES
from run import run_mirror_spot_experiment

SPOT_KINDS = ('r', 't45')


def parse_kind_list(text):
    kinds = parse_comma_list(text)
    unknown = [k for k in kinds if k not in SPOT_KINDS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown kind(s) {unknown!r}; choose from {','.join(SPOT_KINDS)}"
        )
    if not kinds:
        raise argparse.ArgumentTypeError(
            f"need at least one of {','.join(SPOT_KINDS)}"
        )
    return [k for k in SPOT_KINDS if k in kinds]


def configure_parser(ap):
    ap.add_argument(
        'kind',
        type=parse_kind_list,
        metavar='KIND',
        help='comma-separated: r (photoreceptors→L1), t45 (T4a-d/T5a-d→L4/L5); e.g. r,t45',
    )


def mirror_fits(args):
    fits = []
    if 'r' in args.kind:
        fits.append({'mirror_types': ['R1-6', 'R7', 'R8'], 'mirror_fit': 'L1'})
    if 't45' in args.kind:
        t4 = [n for n in READOUT_SUBTYPES if n.startswith('T4')]
        t5 = [n for n in READOUT_SUBTYPES if n.startswith('T5')]
        fits.append({'mirror_types': t4, 'mirror_fit': 'L4'})
        fits.append({'mirror_types': t5, 'mirror_fit': 'L5'})
    return fits


run_mirror_spot_experiment(
    __doc__,
    lambda args: 'add_spot_' + '_'.join(args.kind),
    mirror_fits,
    configure_parser=configure_parser,
)
