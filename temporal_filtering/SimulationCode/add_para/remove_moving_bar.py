"""Experiment: drop T4/T5 subtypes from moving-bar cost (default pool minus positional)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import FiveCol_MedSim_Pytorch as fc
import train
from t4_t5_preference import READOUT_SUBTYPE_ALIASES, READOUT_SUBTYPES, expand_remove_subtypes_list

ap = train.make_training_argparser(__doc__)
valid = ", ".join((*READOUT_SUBTYPE_ALIASES, *READOUT_SUBTYPES))
ap.add_argument(
    "remove_subtypes",
    metavar="SUBTYPES",
    help=f"comma-separated subtypes to drop from moving-bar cost ({valid}); aliases T4, T5",
)
args = ap.parse_args()
try:
    train_kw = train.training_kwargs_from_args(args, script_stem='remove_moving_bar')
except ValueError as exc:
    ap.error(str(exc))

try:
    removed = frozenset(expand_remove_subtypes_list(train.parse_comma_list(args.remove_subtypes)))
    readout_subtypes = [st for st in READOUT_SUBTYPES if st not in removed]
    if not readout_subtypes:
        ap.error(f"removing {sorted(removed)!r} leaves no moving-bar readout subtypes")
except ValueError as exc:
    ap.error(str(exc))

target_list = train_kw['target_list']
bar_targets = [t for t in target_list if t in fc.MOVING_BAR_TARGETS]
if not bar_targets:
    ap.error(
        f"--target must include a moving-bar target "
        f"({', '.join(fc.MOVING_BAR_TARGETS)} or moving_bar)",
    )

moving_bar_bright_stimulus_opts = None
moving_bar_dark_stimulus_opts = None
if 'moving_bar_bright' in bar_targets:
    moving_bar_bright_stimulus_opts = fc.make_moving_bar_bright_stimulus_opts(
        readout_subtypes=readout_subtypes,
    )
if 'moving_bar_dark' in bar_targets:
    moving_bar_dark_stimulus_opts = fc.make_moving_bar_dark_stimulus_opts(
        readout_subtypes=readout_subtypes,
    )

fname, outdir, session = train.run_training(
    **train_kw,
    moving_bar_bright_stimulus_opts=moving_bar_bright_stimulus_opts,
    moving_bar_dark_stimulus_opts=moving_bar_dark_stimulus_opts,
)
for tname in bar_targets:
    pack = session.pack_for(tname)
    print(f'{tname} cost cells:', int(pack.readout_unit.shape[0]))
    print(f'{tname} readout subtypes:', session.train_opts[f'{tname}_stimulus_opts']['readout_subtypes'])
print('done ->', outdir)
