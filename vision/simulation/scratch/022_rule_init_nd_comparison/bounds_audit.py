from pathlib import Path
import json,numpy as np,pandas as pd
O=Path(__file__).resolve().parent;run=Path(json.loads((O/'cases.json').read_text())[1]['run']);opts=json.loads((run/'data/train_opts.json').read_text());cells=pd.read_csv(run/'param.csv').cell.tolist();rows=[]
with np.load(run/'data/best_param.npz') as a:
 fixed=set(opts['params']['a_gt']['modes']['fixed'])
 for i,cell in enumerate(cells):
  if cell not in fixed:rows.append(dict(parameter='a_gt',scope=cell,value=float(a['a_gt'][i]),lo=opts['params']['a_gt']['lo'],hi=opts['params']['a_gt']['hi']))
 for name in ['a_sti_mid','v_th_syn','k_minus_syn']:
  assert np.all(a[name]==a[name].reshape(-1)[0]);spec=opts['params'][name];rows.append(dict(parameter=name,scope='shared',value=float(a[name].reshape(-1)[0]),lo=spec['lo'],hi=spec['hi']))
 for name,spec in opts['point_hh']['gene']['params'].items():
  assert np.all(a[name]==a[name].reshape(-1)[0]);rows.append(dict(parameter=name,scope='shared',value=float(a[name].reshape(-1)[0]),lo=spec['lo'],hi=spec['hi']))
raw=np.load(run/'data/train_point_hh.npy');assert len(rows)==raw.size==62
r=pd.DataFrame(rows);r['at_upper']=np.isclose(r.value,r.hi,atol=1e-6,rtol=0);r['at_lower']=np.isclose(r.value,r.lo,atol=1e-6,rtol=0);r.to_csv(O/'init_trainable_bounds.csv',index=False)
summary=json.loads((O/'init_best/metrics.json').read_text())['parameter_summary'];assert int(r.at_upper.sum())==summary['at_upper_bound']==4;assert int(r.at_lower.sum())==summary['at_lower_bound']==0
folds=opts['point_hh']['gene']['max_fold_changes'];mem=pd.read_csv(O/'init_best/membrane_conductances.csv');effective=[]
for _,row in mem.iterrows():
 for name in ['gNa','gK','gLeak']:
  base=row[name+'_base_nS'];value=row[name+'_trained_nS'];effective.append(dict(parameter=name,scope=row.neuron_type,baseline=base,value=value,fold=value/base,max_fold=folds[name],effective_upper=base*folds[name],fraction_of_upper=value/(base*folds[name])))
syn=pd.read_csv(O/'init_best/synapse_conductances.csv')
for _,row in syn.iterrows():
 base=row.base_max_nS;value=row.trained_max_nS;effective.append(dict(parameter='gS_per_contact',scope=f'{row.pre_type}->{row.post_type} {row.model_nt_type}',baseline=base,value=value,fold=value/base,max_fold=folds['gS'],effective_upper=base*folds['gS'],fraction_of_upper=value/(base*folds['gS'])))
e=pd.DataFrame(effective);e['at_effective_upper']=np.isclose(e.value,e.effective_upper,atol=1e-6,rtol=0);e.to_csv(O/'init_effective_conductance_bounds.csv',index=False)
g=r[r.parameter.str.startswith('gene_')];peak=g.loc[g.value.abs().idxmax()]
result=dict(run=str(run),trainable_count=len(r),hard_upper=r[r.at_upper].to_dict('records'),hard_lower=r[r.at_lower].to_dict('records'),fixed_T4_a_gt=1.,gene_coefficients_at_bounds=int((g.at_upper|g.at_lower).sum()),maximum_absolute_gene_coefficient=dict(parameter=peak.parameter,value=peak.value,bounds=[peak.lo,peak.hi]),effective_upper_hit_count=int(e.at_effective_upper.sum()),near_effective_upper=e[e.fraction_of_upper>=.90].to_dict('records'))
(O/'init_bounds_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
