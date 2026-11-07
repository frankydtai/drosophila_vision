from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
source = (HERE / '4_optimization.before.py').read_text()

def edit(old, new, count=1):
    global source
    assert source.count(old) == count, (old, source.count(old))
    source = source.replace(old, new)

edit('from train.cost import session_cost_part_keys\n', 'from task.mbar.pack import dsi_groups, dsi_values\nfrom train.cost import forward_pack, session_cost_part_keys\n')
edit('    active_device,\n', '    active_device,\n    params_from_z,\n')
edit('_TQDM_REFRESH_INTERVAL = 10\n', '''def _calc_dsi(z, session):
    """Mean raw moving-bar DSI across cell/contrast groups, independent of loss scales."""
    values = []
    with torch.no_grad():
        for pack in session.iter_packs():
            if not any(True for _ in dsi_groups(pack)):
                continue
            params = params_from_z(z, session)
            trace = forward_pack(session, params, pack.i_sti, pack)
            values.extend(float(value.item()) for value in dsi_values(pack, trace).values())
    return sum(values) / len(values) if values else None


def _fmt_dsi(dsi):
    return "" if dsi is None else f"  DSI: {dsi:.4f}"


_TQDM_REFRESH_INTERVAL = 10
''')
edit('                     adam_init=None):', '                     adam_init=None, dsi_fn=None):')
edit('    initial_cost = 1.0 * cost\n', '    dsi = dsi_fn(z) if dsi_fn is not None else None\n    initial_dsi = best_dsi = dsi\n    initial_cost = 1.0 * cost\n')
edit('            initial_parts, summary_only=True,\n        ),', '            initial_parts, summary_only=True,\n        ) + _fmt_dsi(initial_dsi),')
edit('        if cost < best_cost:\n', '        dsi = dsi_fn(z) if dsi_fn is not None else None\n\n        if cost < best_cost:\n')
edit('        if cost < best_cost:\n\n            best_cost = cost\n', '        if cost < best_cost:\n\n            best_dsi = dsi\n            best_cost = cost\n')
edit('                summary_only=True,\n            ),', '                summary_only=True,\n            ) + _fmt_dsi(dsi),')
edit('            final_parts = float_last_parts(task_order) if float_last_parts else None\n', '            final_parts = float_last_parts(task_order) if float_last_parts else None\n            dsi = dsi_fn(z) if dsi_fn is not None else None\n')
edit("            cost = float('nan')\n            final_parts = None\n", "            cost = float('nan')\n            dsi = None\n            final_parts = None\n")
edit('                best_cost = cost\n', '                best_dsi = dsi\n                best_cost = cost\n')
edit("        cost = float('nan')\n        final_parts = None\n", "        cost = float('nan')\n        dsi = None\n        final_parts = None\n")
for kind, metric in [('initial', 'initial_dsi'), ('final', 'dsi'), ('best', 'best_dsi')]:
    parts = 'initial' if kind == 'initial' else kind
    edit(f'        {parts}_parts, summary_only=True,\n    ))', f'        {parts}_parts, summary_only=True,\n    ) + _fmt_dsi({metric}))')
edit('                 adam_init=None):', '                 adam_init=None, dsi_fn=None):')
edit('            adam_init=adam_init if stage_i == 0 else None,\n', '            adam_init=adam_init if stage_i == 0 else None,\n            dsi_fn=dsi_fn,\n')
edit('            adam_init=adam_init,\n', '            adam_init=adam_init,\n            dsi_fn=lambda z: _calc_dsi(z, session),\n')
compile(source, '4_optimization.candidate.py', 'exec')
(HERE / '4_optimization.candidate.py').write_text(source)
paths = [ROOT / 'vision/simulation/conf/config.yaml', ROOT / 'vision/simulation/4_train/3_cost.py', ROOT / 'vision/simulation/run_init_from.slurm']
(HERE / 'protected_hashes.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
