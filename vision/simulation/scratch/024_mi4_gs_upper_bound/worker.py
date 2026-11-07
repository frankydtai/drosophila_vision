import sys,json,math
from pathlib import Path
from response_metrics import O,setup,evaluate
import numpy as np,pandas as pd,torch
import neuron.point_hh as hh
import train.cost as costmod
from neuron.readout import pack_traces
case=json.loads((O/'cases.json').read_text())[int(sys.argv[1])];folder=O/case['label'];folder.mkdir(exist_ok=True)
setup();base_prepare=hh.prepare_training_params
# Runtime-only asymmetric cap: negative gene signals keep the original 1/6 floor.
def capped_prepare(decoded,session):
 p=base_prepare(decoded,session)
 if case['upper_fold']!=6 and case['kind']!='native_symmetric':
  n_ion=len(hh.ION_GENE_LINKS);coeff=torch.stack([decoded[name][0] for name in hh.GENE_PARAM_NAMES]);signal=session.model_context.gene_design['edge_features']@coeff[n_ion:];shape=torch.tanh(signal)
  if case['kind']=='upper_only':shape=shape.clamp_min(0)
  p['gS']=p['gS']*torch.exp((math.log(case['upper_fold'])-math.log(6))*shape)
 return p
hh.prepare_training_params=capped_prepare
original_forward=costmod.forward_pack;original_step=hh.step;original_pre=hh.pre_steady
active=False;tick=0;buffers={};mi4={};mi4_curves={}
def record(state,params,t):
 mask=buffers['edge_mask'];targets=buffers['targets'];g=state['s'][:,mask]*params['gS'][mask];out=g*(state['v'].index_select(-1,targets)-params['e_syn'][mask]);index=targets.expand_as(g)
 buffers['g'][:,t].scatter_add_(1,index,g);buffers['i'][:,t].scatter_add_(1,index,out)
 assert bool(torch.isfinite(g).all()) and bool(((state['s'][:,mask]>=0)&(state['s'][:,mask]<=1)).all())
def pre_capture(*args,**kwargs):
 state=original_pre(*args,**kwargs)
 if active:record(state,args[1],0)
 return state
def step_capture(state,params,*args,**kwargs):
 global tick
 new=original_step(state,params,*args,**kwargs)
 if active:tick+=1;record(new,params,tick)
 return new
hh.pre_steady=pre_capture;hh.step=step_capture

def observed_forward(session,params,i_sti,pack):
 global active,tick,buffers
 full=pack.task=='mbar' and int(i_sti.shape[0])==2
 if full:
  source,target=hh._edge_indices(session,params['gS'].device,params)
  mask=torch.tensor([r['pre_type']=='Mi4' and r['post_type'] in ['T4a','T4b'] and r['model_nt_type']=='GABA' and r['e_syn_mV']<-40 for r in session.model_context.edge_rows],device=session.device)
  n_b,n_t,n_node=i_sti.shape;buffers=dict(edge_mask=mask,targets=target[mask],g=i_sti.new_zeros(n_b,n_t,n_node),i=i_sti.new_zeros(n_b,n_t,n_node));active=True;tick=0
 trace=original_forward(session,params,i_sti,pack)
 if full:
  active=False;assert tick==trace.shape[1]-1
  g=pack_traces(buffers['g'],pack);current=pack_traces(buffers['i'],pack)
  for key in dict.fromkeys(pack.entry_part_keys):
   m=torch.tensor([k==key for k in pack.entry_part_keys],device=trace.device)&(pack.cost_scales>0)
   if not bool(m.any()):continue
   gm=g[m].mean(0);im=current[m].mean(0);baseline=buffers['i'][pack.entry_bs[m],0,pack.entry_nodes[m]].mean();evoked=im-baseline
   nominal_node=trace.new_zeros(trace.shape[-1]);nominal_node.scatter_add_(0,buffers['targets'],params['gS'][buffers['edge_mask']]);nominal=nominal_node[pack.entry_nodes[m]].mean();base_g=buffers['g'][pack.entry_bs[m],0,pack.entry_nodes[m]].mean()
   mi4[key]=dict(nominal_total_gS_nS=float(nominal),peak_weighted_open_fraction=float(gm.max()/nominal),baseline_weighted_open_fraction=float(base_g/nominal),peak_open_conductance_nS=float(gm.max()),peak_outward_current_pA=float(im.max()),peak_evoked_outward_current_pA=float(evoked.max()),evoked_current_integral_pA_ms=float(evoked.sum()*session.delta_ms))
   mi4_curves[key+'__g']=gm.numpy();mi4_curves[key+'__i']=im.numpy();mi4_curves[key+'__evoked_i']=evoked.numpy()
  mi4['source_voltage_summary']=dict(baseline_mean_mV=float(trace[:,0,source[mask].unique()].mean()),maximum_spatial_mean_mV=float(trace[:,:,source[mask].unique()].mean(-1).max()))
  buffers={}
 return trace
costmod.forward_pack=observed_forward
metrics,curves=evaluate(case['run'],gs_fold=case['upper_fold'] if case['kind']=='native_symmetric' else None);ions,syn,summary=metrics.pop('parameter_tables');metrics.update(case=case,parameter_summary=summary,mi4_output=mi4,nominal_per_contact_gS_upper_nS=.3*case['upper_fold'],nominal_per_contact_gS_lower_nS=.3/(6 if case['kind']=='upper_only' else case['upper_fold']))
if case['upper_fold']==6:assert metrics['total']==float(np.load(Path(case['run'])/'data/costs.npy')[0])
pd.DataFrame(ions).to_csv(folder/'membrane_conductances.csv',index=False);syn.to_csv(folder/'synapse_conductances.csv',index=False)
np.savez(folder/'curves.npz',**{key+'__'+label:v for key,pair in curves.items() for label,v in zip(['model','gt'],pair)});np.savez(folder/'mi4_currents.npz',**mi4_curves)
(folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2),flush=True)
