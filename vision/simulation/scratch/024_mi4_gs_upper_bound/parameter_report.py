import numpy as np,pandas as pd,torch
from train.param import params_from_z,schema_clamps

def collect(session,z):
 with torch.no_grad():
  p=params_from_z(z,session);ctx=session.model_context;ions=[]
  for i,cell in enumerate(session.connectome.cells):
   mask=session.connectome.node_cells==i;row=dict(neuron_type=str(cell))
   for name in ['gNa','gK','gLeak']:
    row[name+'_base_nS']=float(ctx.params[name][mask].mean());row[name+'_trained_nS']=float(p[name][mask].mean())
   ions.append(row)
  e=pd.DataFrame(ctx.edge_rows);n=session.connectome.edge_n_syn.numpy();e['base_per_contact_nS']=ctx.params['gS'].numpy()/n;e['trained_per_contact_nS']=p['gS'].numpy()/n
  syn=e.groupby(['pre_type','post_type','model_nt_type','e_syn_mV'],as_index=False).agg(n_edges=('edge_index','size'),n_syn=('n_syn','sum'),base_min_nS=('base_per_contact_nS','min'),base_max_nS=('base_per_contact_nS','max'),trained_min_nS=('trained_per_contact_nS','min'),trained_max_nS=('trained_per_contact_nS','max'))
  clamps=schema_clamps(session.schema,session.sim_dtype);array=z.numpy();bounds=clamps.numpy()
  summary=dict(free_parameters=int(z.numel()),at_lower_bound=int(np.isclose(array,bounds[:,0],atol=1e-6,rtol=0).sum()),at_upper_bound=int(np.isclose(array,bounds[:,1],atol=1e-6,rtol=0).sum()),zero_gS_edges=int((p['gS']==0).sum()),active_gS_edges=int((p['gS']>0).sum()))
 return ions,syn,summary
