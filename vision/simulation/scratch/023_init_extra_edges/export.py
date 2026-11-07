from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[3]/'vision/simulation/0_runs/point_hh'
a=R/'34437673-continued_optimization-loss-282967/data/point_hh_edge_params.csv';b=R/'34466728-init_nd_optimization-loss-994279/data/point_hh_edge_params.csv'
r=pd.read_csv(a);i=pd.read_csv(b)
for key in ['edge_index','src','tar','pre_type','post_type','n_syn']:assert r[key].equals(i[key]),key
mask=(r.gS_nS==0)&(i.gS_nS>0);assert int(mask.sum())==3309
assert set(r.loc[mask,'resolution'])=={'unresolved_zero'} and set(i.loc[mask,'resolution'])=={'unresolved_init'}
e=i.loc[mask,['edge_index','src','tar','pre_type','post_type','n_syn','raw_nt_type','model_nt_type','network_sign','model_sign','e_syn_mV']].copy()
e['rule_context_gS_nS']=r.loc[mask,'gS_nS'];e['init_context_gS_nS']=i.loc[mask,'gS_nS'];e['rule_resolution']=r.loc[mask,'resolution'];e['init_resolution']=i.loc[mask,'resolution'];e['direct_to_T4']=e.post_type.str.startswith('T4');e['T4_to_T4']=e.pre_type.str.startswith('T4')&e.post_type.str.startswith('T4')
assert (e.e_syn_mV==0).all() and (e.model_sign==1).all();assert np.allclose(e.init_context_gS_nS,.3*e.n_syn)
e.to_csv(O/'newly_enabled_edges.csv',index=False)
groups=e.groupby(['pre_type','post_type','model_nt_type','e_syn_mV'],as_index=False).agg(edge_count=('edge_index','size'),contact_count=('n_syn','sum'),init_context_gS_sum_nS=('init_context_gS_nS','sum')).sort_values(['edge_count','pre_type'],ascending=[False,True]);groups.to_csv(O/'by_cell_pair_and_transmitter.csv',index=False)
pre=e.groupby('pre_type',as_index=False).agg(edge_count=('edge_index','size'),contact_count=('n_syn','sum')).sort_values('edge_count',ascending=False);pre.to_csv(O/'by_source_type.csv',index=False)
t=e[e.direct_to_T4];t.to_csv(O/'direct_to_T4_edges.csv',index=False)
tg=groups[groups.post_type.str.startswith('T4')].sort_values(['pre_type','post_type','model_nt_type']);tg.to_csv(O/'direct_to_T4_groups.csv',index=False)
nt=e.groupby(['raw_nt_type','model_nt_type','e_syn_mV'],as_index=False).agg(edge_count=('edge_index','size'),contact_count=('n_syn','sum'))
summary=dict(total_network_edges=len(r),newly_enabled_edge_count=len(e),newly_enabled_contact_count=int(e.n_syn.sum()),direct_to_T4_edges=len(t),T4_to_T4_edges=int(e.T4_to_T4.sum()),by_source_type=pre.to_dict('records'),by_transmitter=nt.to_dict('records'),direct_to_T4_groups=tg.to_dict('records'),context_values_are_baselines_before_gene_multiplier=True,all_newly_enabled_edges_have_fallback_e_syn_mV=0.,all_newly_enabled_edges_have_positive_model_sign=True,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a,b]})
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
protected=json.loads((O.parent/'022_rule_init_nd_comparison/protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in summary['sources'].items())
print(json.dumps(dict(edges=len(e),contacts=int(e.n_syn.sum()),direct_to_T4=len(t),T4_to_T4=int(e.T4_to_T4.sum()),by_source=pre.to_dict('records'),by_nt=nt.to_dict('records'),folder=str(O)),indent=2))
