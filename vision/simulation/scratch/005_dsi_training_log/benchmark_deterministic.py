from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'vision'))
import import_bootstrap
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True)
import train.cost as cost
from train.session import resolve_session
from train.implementation import load_best_node_vals, session_z_from_node_vals
from train.param import schema_clamps

sys.path.insert(0, str(ROOT / 'vision/simulation'))
spec = importlib.util.spec_from_file_location('dsi_candidate_benchmark', HERE / '4_optimization.candidate.py')
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
torch.set_num_threads(1)
run = ROOT / 'vision/simulation/0_runs/point_hh/34430984-run_init_from-n_iter-50-tasks-mbar-contrasts-bright-part_cost_scales-mbar-1,mbar_DSI-0'
opts = json.loads((run / 'data/train_opts.json').read_text())
opts['device'] = 'cuda'
(HERE / 'deterministic_opts.json').write_text(json.dumps(opts, indent=2) + '\n')
# Disable cache writes only in this benchmark's process.
import task.mbar.pack as mbar_pack
build_gt = mbar_pack.build_mbar_gt
mbar_pack.build_mbar_gt = lambda *args, **kwargs: build_gt(*args, **dict(kwargs, use_cache=False))
session = resolve_session(opts)
session, z0 = session_z_from_node_vals(*load_best_node_vals(str(run)), session)
clamps = schema_clamps(session.schema, session.sim_dtype).to(session.device)
print('READY', torch.cuda.get_device_name(0), 'n_z', z0.numel(), flush=True)

def sync():
    torch.cuda.synchronize()

def step(with_dsi):
    z = torch.nn.Parameter(z0.detach().clone())
    optimizer = torch.optim.Adam([z], lr=0.0001)
    sync()
    start = time.perf_counter()
    optimizer.zero_grad()
    parts = cost.calc_cost_parts(z, session)
    value = cost._scaled_cost_from_parts(parts, session)
    value.backward()
    sync()
    after_backward = time.perf_counter()
    dsi = candidate._calc_dsi(z, session) if with_dsi else None
    sync()
    after_dsi = time.perf_counter()
    grad = z.grad.detach().clone()
    optimizer.step()
    with torch.no_grad():
        z.clamp_(clamps[:, 0], clamps[:, 1])
    # Exercise the actual cost-parts and DSI formatting used in each iteration.
    desc = f'Cost: {float(value.detach()):.4f}' + candidate._fmt_cost_parts(
        candidate._float_parts(parts), summary_only=True,
    ) + candidate._fmt_dsi(dsi)
    sync()
    elapsed = time.perf_counter() - start
    result = dict(total_s=elapsed, forward_backward_s=after_backward-start,
                  dsi_s=after_dsi-after_backward, cost=float(value.detach()), dsi=dsi,
                  description=desc, grad_finite=bool(torch.isfinite(grad).all()),
                  grad_sha256=hashlib.sha256(grad.cpu().numpy().tobytes()).hexdigest(),
                  next_z_sha256=hashlib.sha256(z.detach().cpu().numpy().tobytes()).hexdigest())
    del value, parts, z, optimizer, grad
    gc.collect()
    return result

results = dict(source_run=str(run), device=torch.cuda.get_device_name(0), hostname=os.uname().nodename,
               torch_version=torch.__version__, repeats=1, warmups=2, records=[])
for warm in (False, True):
    print('WARMUP', warm, json.dumps(step(warm)), flush=True)
for pair in range(1):
    modes = (False, True) if pair % 2 == 0 else (True, False)
    for enabled in modes:
        record = dict(pair=pair, enabled=enabled, **step(enabled))
        results['records'].append(record)
        (HERE / 'deterministic_results.json').write_text(json.dumps(results, indent=2) + '\n')
        print('MEASURE', json.dumps(record), flush=True)
base = [r['total_s'] for r in results['records'] if not r['enabled']]
new = [r['total_s'] for r in results['records'] if r['enabled']]
results['summary'] = dict(baseline_mean_s=statistics.mean(base), baseline_sd_s=0.0,
    new_mean_s=statistics.mean(new), new_sd_s=0.0,
    extra_mean_s=statistics.mean(new)-statistics.mean(base),
    extra_percent=100*(statistics.mean(new)/statistics.mean(base)-1),
    dsi_mean_s=statistics.mean(r['dsi_s'] for r in results['records'] if r['enabled']))
results['same_cost'] = len({r['cost'] for r in results['records']}) == 1
results['same_gradients'] = len({r['grad_sha256'] for r in results['records']}) == 1
results['same_adam_update'] = len({r['next_z_sha256'] for r in results['records']}) == 1
(HERE / 'deterministic_results.json').write_text(json.dumps(results, indent=2) + '\n')
print('SUMMARY', json.dumps(results['summary']), flush=True)
