from pathlib import Path
import json,numpy as np
from response_metrics import setup,evaluate
O=Path(__file__).resolve().parent;setup();old=json.loads((O/'old_mapping_metrics.json').read_text());new=json.loads((O/'new_mapping_metrics.json').read_text());rows=[]
for name in ['screen_cases.json','refine_cases.json']:
 p=O/name
 if not p.exists():continue
 for case in json.loads(p.read_text()):
  folder=O/case['label'];d=json.loads((folder/'progress.json').read_text());assert d.get('complete'),folder
  run=folder/'run';opts=json.loads((run/'data/train_opts.json').read_text());assert opts['point_hh']['gene']['max_fold_changes']['gS']==10.0 and opts['point_hh']['g_context_mode']=='init'
  saved=float(np.load(run/'data/costs.npy')[0]);assert saved==d['best_cost']
  row=dict(label=case['label'],run=str(run),cost=saved,nd_sum=d['best_nd'],adam_iter=d['selected_adam_iter'],case=case,stop_reason=d['stop_reason'])
  metrics,curves=evaluate(run);ions,syn,param_summary=metrics.pop('parameter_tables');assert metrics['total']==saved;row['metrics']=metrics
  syn.to_csv(folder/'effective_synapse_conductances.csv',index=False)
  import pandas as pd
  pd.DataFrame(ions).to_csv(folder/'effective_membrane_conductances.csv',index=False)
  np.savez(folder/'response_curves.npz',**{key+'__'+label:v for key,pair in curves.items() for label,v in zip(['model','gt'],pair)})
  row['ND_response_improved']=all(metrics['groups'][k]['nd_peak_mV']<b['nd_peak_mV'] and metrics['groups'][k]['nd_pd_ratio']<b['nd_pd_ratio'] for k,b in old['groups'].items())
  row['gS_range_valid']=bool((syn.trained_min_nS>=.03-1e-6).all() and (syn.trained_max_nS<=3.+1e-6).all());assert row['gS_range_valid']
  assert param_summary['active_gS_edges']==27824 and param_summary['zero_gS_edges']==0
  rows.append(row)
from snapshot_candidates import add_candidates
add_candidates(O,rows,old)
(O/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')
feasible=[r for r in rows if r['ND_response_improved'] and r['cost']<=new['total']];assert feasible,'No saved candidate retains both ND peak and ratio improvements; inspect candidate metrics.'
winner=min(feasible,key=lambda d:d['cost']);(O/'selected.json').write_text(json.dumps(winner,indent=2)+'\n');(O/'mbar_baseline.json').write_text(json.dumps(dict(initial_cost=old['total'],new_mapping_initial_cost=new['total']),indent=2)+'\n');print(json.dumps(winner,indent=2),flush=True)
