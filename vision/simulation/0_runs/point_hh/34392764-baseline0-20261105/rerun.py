"""Execution-only baseline override; run with vision/.venv/bin/python. No shared files edited."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parent
SIM = OUT.parents[2]
sys.path.insert(0, str(SIM))
os.chdir(SIM)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import import_bootstrap
import torch
torch.set_num_threads(1)

# Reuse the exact resolved configuration of the requested source run.
source_log = SIM / '0_logs/34392764.log'
config = ast.literal_eval(source_log.read_text().splitlines()[0].removeprefix('config: '))
config['i_sti']['baseline'] = 0.0
config['run_name'] = OUT.name
config['run_path'] = 'point_hh/' + OUT.name
config['figure_tasks'] = ['spread', 'mbar', 'sbar']
(OUT / 'resolved_config.json').write_text(json.dumps(config, indent=2) + '\n')
config_path = SIM / 'conf/config.yaml'
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()

# Patch only this Python process, before the task modules bind the helper.
from task.spread import sti_spec
original_baseline = sti_spec.i_baseline_from_i_sti
def task_baseline(i_sti):
    return float(i_sti['baseline']) if 'baseline' in i_sti else original_baseline(i_sti)
sti_spec.i_baseline_from_i_sti = task_baseline
for module in list(sys.modules.values()):
    if module is not None and getattr(module, 'i_baseline_from_i_sti', None) is original_baseline:
        module.i_baseline_from_i_sti = task_baseline
assert task_baseline({'bright': 40, 'dark': 0}) == 20
assert task_baseline(config['i_sti']) == 0

# A dark=0 / baseline=0 control has no current-defined bar onset. Use a
# virtual 1 pA bar solely to construct geometry/readout windows, then replace
# the simulation input with exactly zero. Keep caches untouched.
from task.mbar import pack as mbar_pack
original_signals = mbar_pack.build_mbar_signals
def timing_signals(*args, **kwargs):
    kwargs['use_cache'] = False
    if float(kwargs['i_sti']) == float(kwargs['i_baseline']):
        kwargs['i_sti'] = float(kwargs['i_baseline']) + 1.0
    return original_signals(*args, **kwargs)
mbar_pack.build_mbar_signals = timing_signals
original_gt = mbar_pack.build_mbar_gt
def control_gt(*args, **kwargs):
    result = original_gt(*args, **kwargs)
    if float(kwargs['i_sti']) == float(kwargs['i_baseline']):
        result.i_sti.fill(float(kwargs['i_baseline']))
        assert (result.i_sti == 0).all()
    return result
mbar_pack.build_mbar_gt = control_gt

from config import resolve_run_kwargs
from run import run_train_and_plot
kwargs = resolve_run_kwargs(config)
kwargs['run_dir'] = str(OUT)
kwargs['fp'] = 64  # Preserve original precision even on CPU.
print('Execution-only overrides: baseline=0 pA; bright=40 pA; dark=0 pA; fp=64', flush=True)
print('Output:', OUT, flush=True)
_, _, session = run_train_and_plot(**kwargs)
for contrast, pack in session.packs['spread'].items():
    assert torch.count_nonzero(pack.i_sti[:, :pack.t_onset, :]) == 0
    if contrast == 'dark':
        assert torch.count_nonzero(pack.i_sti) == 0
expected = [f'{task}_{scope}_v.png' for task in config['figure_tasks'] for scope in ['gt', 'all']]
assert all((OUT / name).stat().st_size > 0 for name in expected)
assert hashlib.sha256(config_path.read_bytes()).hexdigest() == config_hash
(OUT / 'verification.json').write_text(json.dumps({
    'source_run': '34392764-run-experiment-point_hh',
    'baseline_pA': 0, 'bright_pA': 40, 'dark_pA': 0,
    'figure_tasks': config['figure_tasks'], 'figures': expected,
    'original_config_sha256': config_hash, 'original_config_unchanged': True,
    'dark_mbar_timing': 'virtual bar for geometry only; actual simulation current is zero',
}, indent=2) + '\n')
print('VERIFIED: all six figures exist; baseline/dark input zero; original config unchanged.', flush=True)
