from pathlib import Path
import sys,json
O=Path(__file__).resolve().parent;C=O/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np,torch
import train
import train.implementation as impl
import train.cost as costmod
from train.param import params_from_z
from neuron.readout import pack_traces
import task.mbar.pack as packmod

def setup():
 torch.set_num_threads(1)
 build=packmod.build_mbar_gt
 packmod.build_mbar_gt=lambda *a,**kw:build(*a,**dict(kw,use_cache=False))

def evaluate(run, snapshot=None, override=None):
 run=Path(run);opts=json.loads((run/'data/train_opts.json').read_text());opts['device']='cpu';session=train.resolve_session(opts)
 session,z=impl.session_z_from_node_vals(*impl.load_best_node_vals(str(run)),session)
 if snapshot is not None:
  with np.load(snapshot) as state:z=torch.tensor(state['z'],dtype=session.sim_dtype,device=session.device)
 if override:
  import copy
  from dataclasses import replace
  import neuron.point_hh as hh
  original=session.model_context;other_opts=copy.deepcopy(opts)
  original_mode=original.provenance['g_context_mode'];other_opts['point_hh']['g_context_mode']='rule' if original_mode=='init' else 'init'
  other=hh.build_context(session.connectome,other_opts)
  params=dict(original.params)
  if override in ['membrane','all']:
   for name in ['gNa','gK','gLeak']:params[name]=other.params[name].clone()
  if override in ['synapse','all']:params['gS']=other.params['gS'].clone()
  if override=='close_unresolved':params['gS']=original.params['gS']*(other.params['gS']>0)
  if override=='restore_Mi4_Mi9_T4':
   mask=torch.tensor([row['pre_type'] in ['Mi4','Mi9'] and row['post_type'].startswith('T4') and row['e_syn_mV']<-40 for row in original.edge_rows],device=session.device)
   params['gS']=torch.where(mask,other.params['gS'],original.params['gS'])
  session=replace(session,model_context=replace(original,params=params))
 with torch.no_grad():
  parts=costmod.calc_cost_parts(z,session);total=float(costmod.calc_cost(z,session,parts=parts))
  metrics=dict(total=total,parts={k:float(v) for k,v in parts.items()},groups={});curves={}
  for pack in session.iter_packs():
   if pack.task!='mbar':continue
   params=params_from_z(z,session);trace=costmod.forward_pack(session,params,pack.i_sti,pack);a,bias=costmod.gt_affine_from_pack(params,pack,session)
   readout=pack_traces(trace,pack)-bias[:,None]
   for key in dict.fromkeys(pack.entry_part_keys):
    mask=torch.tensor([k==key for k in pack.entry_part_keys],device=readout.device)&(pack.cost_scales>0)
    if bool(mask.any()):
     curves[key]=(readout[mask].mean(0).numpy(),(a[:,None]*pack.gts)[mask].mean(0).numpy())
   stops=pack.dsi_t0s+pack.dsi_n_ts;before=(pack.dsi_n_ts-1)//4;first=pack.dsi_t0s+before
   offsets=torch.arange(-int(before.max()),int((pack.dsi_n_ts-1-before).max())+1,device=trace.device)
   ts=first[:,None]+offsets[None,:];valid=(ts>=pack.dsi_t0s[:,None])&(ts<stops[:,None])
   response=trace[pack.entry_bs[:,None],ts.clamp(0,trace.shape[1]-1),pack.entry_nodes[:,None]]-bias[:,None]
   expected=packmod.dsi_peak_values(pack,trace,bias)
   for key,pd,nd in packmod.dsi_groups(pack):
    peaks={label:float(response[mask].mean(0)[valid[mask].all(0)].amax().clamp_min(0)) for label,mask in [('PD',pd),('ND',nd)]}
    peak_dsi=(peaks['PD']-peaks['ND'])/(peaks['PD']+peaks['ND']+1e-8)
    assert abs(peak_dsi-float(expected[key]))<1e-6
    metrics['groups'][key]=dict(pd_peak_mV=peaks['PD'],nd_peak_mV=peaks['ND'],nd_pd_ratio=peaks['ND']/max(peaks['PD'],1e-8),dsi_peak=peak_dsi)
 from parameter_report import collect
 metrics["parameter_tables"]=collect(session,z)
 return metrics,curves

