"""Evaluate matched 0/300-iteration runs using the repository's exact cost readout."""
from pathlib import Path
import csv
import json
import sys

SIM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIM))
import import_bootstrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import figure.plot as plot
from task.mbar.pack import dsi_values
from neuron.readout import pack_traces
from train.param import params_from_z
from train.cost import forward_pack, gt_affine_from_pack, _pack_cost_parts_from_v_readout, _weighted_mean_cost_parts

OUT = Path(__file__).resolve().parent
CELLS = ['T4a', 'T4b', 'T4c', 'T4d']
SOURCE = next((SIM/'0_runs/hp_lp').glob('34381383-*/data/best_param.npz'))
RUNS = {
    '0': SIM/'0_runs/hp_lp/34382310-t4abcd-from-34381383-iter-0',
    '300': SIM/'0_runs/hp_lp/34382311-t4abcd-from-34381383-iter-300',
}


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else None


def evaluate(label, run):
    session, z, saved_cost = plot.load_best(str(run))
    pack = session.packs['mbar']['bright']
    params = params_from_z(z, session)
    with torch.no_grad():
        trace = forward_pack(session, params, pack.i_sti, pack)
        readout = pack_traces(trace, pack).cpu().numpy()
        dsis = {k: float(v) for k,v in dsi_values(pack,trace).items()}
        a, bias = gt_affine_from_pack(params, pack, session)
        target = (a[:, None] * pack.gts + bias[:, None]).cpu().numpy()
        bias = bias.cpu().numpy()
        parts = {k: float(v) for k, v in _weighted_mean_cost_parts(
            _pack_cost_parts_from_v_readout(pack, session, a, torch.as_tensor(bias,device=a.device),
                                           torch.as_tensor(readout,device=a.device),trace=trace), session).items()}
        assert np.isclose(sum(parts.values()), saved_cost, rtol=1e-4, atol=.1), (sum(parts.values()), saved_cost)
    weights = pack.cost_scales.cpu().numpy()
    entry_nodes = pack.entry_nodes.cpu().numpy()
    us, vs = np.asarray(session.connectome.us), np.asarray(session.connectome.vs)
    center = (us[entry_nodes] == 0) & (vs[entry_nodes] == 0)
    t = np.arange(readout.shape[1])*session.delta_ms - 324.0
    post = t >= 0
    pre = t < 0
    keys = list(pack.entry_part_keys)
    metrics, curves = [], {}
    for key in dict.fromkeys(keys):
        if key not in parts:
            continue
        cell, direction = key.split('_')[-2:]
        for scope in ('cost_entries', 'center_hex'):
            mask = np.array([k == key for k in keys]) & (weights > 0)
            if scope == 'center_hex':
                mask &= center
            if not mask.any():
                continue
            pred = np.average(readout[mask], weights=weights[mask], axis=0)
            gt = np.average(target[mask], weights=weights[mask], axis=0)
            b = float(np.average(bias[mask], weights=weights[mask]))
            pred_delta = pred - np.mean(pred[pre])
            gt_delta = gt - np.mean(gt[pre])
            row = dict(iterations=int(label), scope=scope, cell=cell, pd_nd=direction,
                       part_key=key, n_entries=int(mask.sum()),
                       official_cost_contribution=parts[key] if scope == 'cost_entries' else None,
                       rmse_mv=float(np.sqrt(np.mean((pred-gt)**2))),
                       response_rmse_mv=float(np.sqrt(np.mean((pred[post]-gt[post])**2))),
                       delta_response_rmse_mv=float(np.sqrt(np.mean((pred_delta[post]-gt_delta[post])**2))),
                       response_correlation=corr(pred[post], gt[post]),
                       baseline_error_mv=float(np.mean(pred[pre]-gt[pre])),
                       peak_delta_mv=float(np.max(pred_delta[post])),
                       gt_peak_delta_mv=float(np.max(gt_delta[post])),
                       trough_delta_mv=float(np.min(pred_delta[post])),
                       peak_time_ms=float(t[post][np.argmax(pred_delta[post])]),
                       gt_peak_time_ms=float(t[post][np.argmax(gt_delta[post])]))
            metrics.append(row)
            curves[(cell, direction, scope)] = (t, pred-b, gt-b)
    return metrics, curves, dict(saved_cost=saved_cost, recalculated_cost=sum(parts.values()), parts=parts, raw_pd_minus_nd_sum=dsis)


def main():
    torch.set_num_threads(1)
    all_metrics, curves, summary = [], {}, {}
    for label, run in RUNS.items():
        print(f'Evaluating {label}: {run}', flush=True)
        rows, curve, info = evaluate(label, run)
        all_metrics.extend(rows)
        curves[label] = curve
        with np.load(run/'data/best_adam.npz', allow_pickle=True) as f:
            info['best_adam_iteration'] = int(f['iter'])
        summary[label] = info
        save_csv(OUT/f'metrics_{label}.csv', rows)
        print(json.dumps(info, indent=2), flush=True)
    save_csv(OUT/'fit_metrics.csv', all_metrics)
    for label in RUNS:
        summary[label]['cells'] = {}
        for cell in CELLS:
            rows = [r for r in all_metrics if r['iterations'] == int(label) and r['cell'] == cell and r['scope'] == 'cost_entries']
            summary[label]['cells'][cell] = dict(
                contribution=sum(r['official_cost_contribution'] for r in rows),
                rmse_mv=float(np.sqrt(np.mean([r['rmse_mv']**2 for r in rows]))),
                response_rmse_mv=float(np.sqrt(np.mean([r['response_rmse_mv']**2 for r in rows]))),
                response_correlation={r['pd_nd']:r['response_correlation'] for r in rows},
                peak_delta_mv={r['pd_nd']:r['peak_delta_mv'] for r in rows},
            )
    for scope in ('cost_entries', 'center_hex'):
        fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey='row')
        for i, cell in enumerate(CELLS):
            for j, pdnd in enumerate(('PD', 'ND')):
                ax = axes[i, j]
                for label, color in (('0', '#3569b0'), ('300', '#da7426')):
                    t, pred, gt = curves[label][(cell, pdnd, scope)]
                    if label == '0':
                        ax.plot(t, gt, color='black', lw=1.8, label='GT')
                    ax.plot(t, pred, color=color, lw=1.5, label=f'{label} iter')
                ax.axvline(0, color='gray', lw=.8, ls=':')
                ax.axhline(0, color='gray', lw=.6)
                ax.set_title(f'{cell} {pdnd}')
                ax.set_xlim(-150, 576)
                ax.grid(alpha=.15)
                if j == 0:
                    ax.set_ylabel('Vm - cell onset bias (mV)')
                if i == 3:
                    ax.set_xlabel('Time from local stimulus onset (ms)')
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(f'T4a/b/c/d from 34381383: {scope.replace("_", " ")}', fontsize=13)
        fig.tight_layout(rect=(0,0,1,.97))
        fig.savefig(OUT/f'comparison_{scope}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4))
    run = RUNS['300']
    costs = np.load(run/'data/best_costs.npy')
    ax.plot(costs, color='black', label='Total')
    with np.load(run/'data/best_costs_by_part.npz') as f:
        for cell in CELLS:
            ks = [k for k in f.files if f'_{cell}_' in k and not k.startswith('mbar_DSI')]
            ax.plot(np.sum([f[k] for k in ks],axis=0), label=cell)
    ax.set(xlabel='Iteration', ylabel='Objective contribution', yscale='log', title='300-iteration run: evaluated training losses')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/'loss_by_cell.png',dpi=160)
    plt.close(fig)
    params = {}
    for label, path in [('source', SOURCE)] + [(k, v/'data/best_param.npz') for k,v in RUNS.items()]:
        with np.load(path,allow_pickle=True) as f:
            params[label] = {k: f[k].copy() for k in f.files}
    param_rows = []
    zero_diff = {}
    for key, src in params['source'].items():
        if key in ('cells','pairs'):
            continue
        zero = params['0'][key]
        final = params['300'][key]
        assert src.shape == zero.shape == final.shape, key
        zero_diff[key] = float(np.max(np.abs(src-zero)))
        labels = (params['source']['pairs'] if key.startswith('syn_strength') else
                  params['source']['cells'] if src.size == len(params['source']['cells']) else ['shared'])
        for name, a, b in zip(labels, zero.ravel(), final.ravel()):
            param_rows.append(dict(parameter=key,cell=str(name),source_0iter=float(a),iter300=float(b),
                                   delta=float(b-a),percent_change=float((b-a)/abs(a)*100) if abs(a)>1e-8 else None))
    save_csv(OUT/'parameter_changes.csv',param_rows)
    summary['max_parameter_difference_source_to_0iter'] = zero_diff
    summary['parameter_families'] = {k:dict(n_changed=sum(abs(r['delta'])>1e-5 for r in param_rows if r['parameter']==k),
                                          max_abs_delta=max(abs(r['delta']) for r in param_rows if r['parameter']==k)) for k in zero_diff}
    # Derived baselines and mean voltages are kept separate from fitted parameters.
    derived_rows = []
    for label, run in RUNS.items():
        with (run/'param.csv').open() as f:
            for row in csv.DictReader(f):
                derived_rows.append(dict(iterations=int(label), **row))
    save_csv(OUT/'exported_parameter_tables.csv',derived_rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SUMMARY', json.dumps(summary, indent=2),flush=True)


if __name__ == '__main__':
    main()
