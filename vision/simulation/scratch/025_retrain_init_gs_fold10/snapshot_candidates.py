from pathlib import Path
import json,shutil
import numpy as np,torch,pandas as pd
import train
import train.cost as costmod
import train.implementation as impl
import train.optimization as optimod
from parameter_report import collect

def add_candidates(O,rows,old):
 probes=json.loads((O/'early_response_probe/metrics.json').read_text())
 for label,metrics in probes.items():
  if not all(metrics['groups'][k]['nd_peak_mV']<b['nd_peak_mV'] and metrics['groups'][k]['nd_pd_ratio']<b['nd_pd_ratio'] for k,b in old['groups'].items()):continue
  folder=O/('early_'+label);folder.mkdir(exist_ok=True);state_path=O/'early_response_probe'/(label+'.npz');shutil.copy2(state_path,folder/'best_state.npz')
  source_progress=json.loads((O/label/'progress.json').read_text());case=source_progress['case'];source=Path(case['source']);opts=json.loads((source/'data/train_opts.json').read_text());opts['device']='cpu';opts['point_hh']['gene']['max_fold_changes']['gS']=10.
  session=train.resolve_session(opts);session,_z,_a=impl.load_init_z(str(source),session,resume_adam=False)
  with np.load(state_path) as a:
   z=torch.tensor(a['z'],dtype=session.sim_dtype);adam={'exp_avg':a['exp_avg'].astype(np.float64),'exp_avg_sq':a['exp_avg_sq'].astype(np.float64),'iter':int(a['iter'])};expected=float(a['cost'])
  with torch.no_grad():parts=costmod.calc_cost_parts(z,session);cost=float(costmod.calc_cost(z,session,parts=parts))
  assert cost==expected==metrics['total'];parts_float={k:float(v) for k,v in parts.items()};n=adam['iter'];history=source_progress['records'][:n]
  result=optimod.TrainResult(run_params=z.numpy()[None,:].astype(np.float64),final_costs=np.array([cost]),costs=np.array([r['cost'] for r in history]),costs_by_part={k:np.array([r['parts'][k] for r in history]) for k in parts_float},final_costs_by_part={k:np.array([v]) for k,v in parts_float.items()},run_adams=(adam,))
  run=folder/'run';impl.save_train_data('train_point_hh.npy',str(run),session,result)
  prov=json.loads((O/label/'run/data/provenance.json').read_text());prov.update(selected_cost=cost,selected_update=n,selected_adam_iter=n,selection='verified_early_snapshot_with_both_ND_peaks_and_ratios_improved',snapshot_origin=str(state_path));(run/'data/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
  ions,syn,summary=collect(session,z);syn.to_csv(folder/'effective_synapse_conductances.csv',index=False);pd.DataFrame(ions).to_csv(folder/'effective_membrane_conductances.csv',index=False)
  # Curves are reproduced for plots from the saved native fold-10 model.
  from response_metrics import evaluate
  fresh,curves=evaluate(run);fresh.pop('parameter_tables');assert fresh['total']==cost
  np.savez(folder/'response_curves.npz',**{key+'__'+part:v for key,pair in curves.items() for part,v in zip(['model','gt'],pair)})
  rows.append(dict(label=folder.name,run=str(run),cost=cost,nd_sum=sum(v for k,v in parts_float.items() if k.endswith('_ND')),adam_iter=n,case=case,metrics=fresh,ND_response_improved=True,gS_range_valid=bool((syn.trained_min_nS>=.03-1e-6).all() and (syn.trained_max_nS<=3.+1e-6).all()),stop_reason='snapshot_from_completed_case'))
