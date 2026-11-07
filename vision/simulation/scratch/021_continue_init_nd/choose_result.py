from pathlib import Path
import json,numpy as np
from response_metrics import setup,evaluate
O=Path(__file__).resolve().parent;setup()
base=json.loads((O/'source_metrics.json').read_text());rows=[]
for name in ['screen_cases.json','refine_cases.json','peak_cases.json']:
 p=O/name
 if not p.exists():continue
 for case in json.loads(p.read_text()):
  folder=O/case['label'];d=json.loads((folder/'progress.json').read_text());assert d.get('complete'),folder
  run=folder/'run';opts=json.loads((run/'data/train_opts.json').read_text());assert opts['point_hh']['g_context_mode']=='init'
  saved=float(np.load(run/'data/costs.npy')[0]);assert saved==d['best_cost']
  row=dict(label=case['label'],run=str(run),cost=saved,nd_sum=d['best_nd'],adam_iter=d['selected_adam_iter'],case=case)
  if saved<base['total'] and all(d['best_parts'][k]<v for k,v in base['parts'].items() if k.endswith('_ND')):
   metrics,curves=evaluate(run);assert metrics['total']==saved
   row['metrics']=metrics;row['response_improved']=all(metrics['groups'][k]['nd_peak_mV']<g['nd_peak_mV'] for k,g in base['groups'].items());row['both_ratios_improved']=all(metrics['groups'][k]['nd_pd_ratio']<g['nd_pd_ratio'] for k,g in base['groups'].items())
   np.savez(folder/'response_curves.npz',**{key+'__'+label:v for key,pair in curves.items() for label,v in zip(['model','gt'],pair)})
  else:row['response_improved']=False
  rows.append(row)
(O/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')
feasible=[r for r in rows if r['response_improved']];assert feasible,'No candidate improves both ND peaks, ND part losses, and original total; inspect all_results.json.'
preferred=[r for r in feasible if r.get('both_ratios_improved')];feasible=preferred or feasible
winner=min(feasible,key=lambda d:(d['nd_sum'],d['cost']));winner['selection_tradeoff']='both_ND_PD_ratios_improved' if winner.get('both_ratios_improved') else 'ND_peaks_and_total_lower_but_at_least_one_ND_PD_ratio_did_not_improve'
(O/'selected.json').write_text(json.dumps(winner,indent=2)+'\n');(O/'mbar_baseline.json').write_text(json.dumps(dict(initial_cost=base['total']),indent=2)+'\n')
print(json.dumps(winner,indent=2),flush=True)
