"""Exact downstream replay and causal restoration of transmitted upstream signals."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json,time,gc,hashlib
import numpy as np
import torch
import prepare as ref
from train.param import override_val_from
from train.cost import _pack_cost_parts_from_v_readout,_weighted_mean_cost_parts,gt_affine_from_pack,pack_traces
from task.mbar.pack import dsi_values
OUT=Path(__file__).resolve().parent
import figure.mbar as figure_mbar
_original_alignment=figure_mbar.mbar_session_t0_grids
_alignment_cache={}
def _cached_alignment(session,specs,cost_radius,n_t,**kwargs):
 key=(id(session.connectome),tuple((x.token,x.speed_deg_over_s) for x in specs),cost_radius,n_t,tuple(sorted(kwargs.items())))
 if key not in _alignment_cache:_alignment_cache[key]=_original_alignment(session,specs,cost_radius,n_t,**kwargs)
 return _alignment_cache[key]
figure_mbar.mbar_session_t0_grids=_cached_alignment
H,Z,BASE=ref.load(ref.HH);PACK=H.packs['mbar']['bright'];NC=H.connectome.node_cells;CELLS=list(H.connectome.cells);SRC=H.connectome.conn.source_idxs;TAR=H.connectome.conn.target_idxs
TARGET=torch.isin(NC,torch.tensor([CELLS.index('T4a'),CELLS.index('T4b')]));NONTARGET=~TARGET;EI=torch.nonzero(TARGET[TAR]).flatten();ESRC=SRC[EI];ETAR=TAR[EI];EPRE=NC[ESRC]
assert (BASE['gS'][TARGET[SRC]]==0).all()
proof=json.loads((OUT/'conditional_graph_proof.json').read_text());HP_DELTA=torch.from_numpy(np.load(OUT/'HPLP_matched_full.npy')-np.load(OUT/'HPLP_matched_background.npy'))
DATA={};ONSET=int(PACK.t_onset);NT=PACK.i_sti.shape[1];OLD_RELEASE=ref.hh._release_step

def load_data(label):
 if label not in DATA:
  v=torch.from_numpy(np.load(OUT/(label+'_full.npy')));bg=torch.from_numpy(np.load(OUT/(label+'_background.npy')));DATA.clear();DATA[label]=(v,v-bg)
 return DATA[label]


def make(case):
 p={k:(v.clone() if torch.is_tensor(v) else v) for k,v in BASE.items()}
 if case.get('k_label','HH_saved')=='HH_k001':p['k_minus_syn'][:]=.01
 mask=(NC[SRC]==CELLS.index('Mi9'))&TARGET[TAR]
 if case.get('mi9_threshold') is not None:p['v_th_syn'][mask]=case['mi9_threshold']
 if case.get('mi9_gS_gain') is not None:p['gS'][mask]*=case['mi9_gS_gain']
 return p


def hook(case,clock,hh_delta,edge_ids):
 es=SRC[edge_ids];precell=NC[es];restore_mi9=precell==CELLS.index('Mi9')
 neg=torch.isin(precell,torch.tensor([CELLS.index(c) for c in case.get('negative_sources',[])]))
 freeze=torch.isin(precell,torch.tensor([CELLS.index(c) for c in case.get('freeze_sources',[])]))
 def release(s,pre,theta,delta,kminus,dt):
  step=clock['t']
  if step>=ONSET:
   dose=case.get('dose',1.);j=step-1-int(case.get('shift_ms',0));current=step-1
   if j<ONSET:hp=torch.zeros_like(pre)
   else:
    hp=HP_DELTA[:,min(j,NT-1),es]
    if case.get('swap_directions'):hp=hp.flip(0)
   dd=hh_delta[:,current,es];pre=pre.clone()
   if case.get('restore_mi9'):pre[:,restore_mi9]+=dose*(hp[:,restore_mi9].clamp_max(0)-dd[:,restore_mi9])
   if bool(neg.any()):pre[:,neg]+=dose*(hp[:,neg].clamp_max(0)-dd[:,neg].clamp_max(0))
  out=OLD_RELEASE(s,pre,theta,delta,kminus,dt)
  if step>=ONSET:
   if bool(freeze.any()):out[:,freeze]=clock['s_anchor'][:,freeze]
   if case.get('mi9_zero'):out[:,restore_mi9]=0.
  return out
 return release

@torch.no_grad()
def conditional(case):
 label=case.get('k_label','HH_saved');vref,hh_delta=load_data(label);p=make(case)
 state=ref.hh.pre_steady(H,p,2,i_sti=PACK.i_sti[:,:1]);assert torch.allclose(state['v'][:,NONTARGET],vref[:,0,NONTARGET],rtol=2e-6,atol=2e-5)
 state['s']=state['s'][:,EI].clone();prepared=ref.hh.prepare_params(p,PACK.i_sti[:,0],H)
 sub={k:prepared[k] for k in ref.hh._MEMBRANE_PARAM_NAMES};sub.update({k:prepared[k][EI] for k in ref.hh._SYNAPSE_PARAM_NAMES});sub.update(_point_hh_prepared=True,_source_idx=ESRC,_target_idx=ETAR)
 output=vref.clone();output[:,0,TARGET]=state['v'][:,TARGET];clock={'t':0,'s_anchor':None};mode=bool(case.get('restore_mi9') or case.get('negative_sources') or case.get('freeze_sources') or case.get('mi9_zero'))
 mi9=EPRE==CELLS.index('Mi9');gmi9=torch.empty(2,NT,int(TARGET.sum()));zero=torch.zeros_like(PACK.i_sti[:,0])
 # All modified postsynaptic T4 nodes have no external current and no effective
 # outgoing edges. Their complement therefore follows the saved HH trajectory.
 assert torch.all(PACK.i_sti[:,:,TARGET]==0)
 def record_g(t):
  g=torch.zeros_like(state['v']);g.scatter_add_(1,ETAR[mi9].expand(2,-1),state['s'][:,mi9]*sub['gS'][mi9]);gmi9[:,t]=g[:,TARGET]
 record_g(0)
 if mode:ref.hh._release_step=hook(case,clock,hh_delta,EI)
 try:
  for ti in range(1,NT):
   clock['t']=ti
   if ti==ONSET:clock['s_anchor']=state['s'].clone()
   state=ref.hh.step(state,sub,zero,H,delta_ms=1.);state['v'][:,NONTARGET]=vref[:,ti,NONTARGET];output[:,ti,TARGET]=state['v'][:,TARGET];record_g(ti)
   if ti%500==0:assert all(torch.isfinite(a).all() for a in state.values())
 finally:ref.hh._release_step=OLD_RELEASE
 assert torch.isfinite(output).all();return output,p,gmi9

@torch.no_grad()
def evaluate(case,v,p,gmi9):
 override_val_from(p,H,onset_trace=v,t_onset=ONSET);metrics,curves=ref.cost_and_curves(H,p,PACK,v)
 zero_sd=replace(H,train_opts={**H.train_opts,'a_lsd':0.})
 parts=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(PACK,zero_sd,*gt_affine_from_pack(p,PACK,zero_sd),pack_traces(v,PACK),trace=v),zero_sd)
 mean_cost=sum(float(a) for a in parts.values());metrics['mean_only_cost']=mean_cost;metrics['sd_band_cost']=metrics['cost']-mean_cost
 dv=v-v[:,ONSET-1:ONSET,:];metrics['dsi_baseline_subtracted']={k:float(a) for k,a in dsi_values(PACK,dv).items()};metrics['tonic_T4_mV']={c:float(v[:,ONSET-1,NC==CELLS.index(c)].mean()) for c in ['T4a','T4b']}
 metrics['tonic_Mi9_conductance_nS']={c:float(gmi9[:,ONSET-1,(NC[TARGET]==CELLS.index(c))].mean()) for c in ['T4a','T4b']}
 metrics['case']=case;np.savez_compressed(OUT/'curves'/(case['name']+'.npz'),**curves)
 return metrics


def cases_primary():
 cases=[dict(name='saved_baseline'),dict(name='saved_repeat'),dict(name='zero_dose',negative_sources=['Mi1','Tm3'],restore_mi9=True,dose=0.),dict(name='Mi9_frozen',freeze_sources=['Mi9']),dict(name='Mi4_frozen',freeze_sources=['Mi4']),dict(name='Mi9_zero',mi9_zero=True)]
 for dose in [.5,1.,1.5]:cases.append(dict(name=f'excitatory_negative_{dose:g}',negative_sources=['Mi1','Tm3'],dose=dose))
 cases += [dict(name='Mi1_negative',negative_sources=['Mi1']),dict(name='Tm3_negative',negative_sources=['Tm3']),dict(name='Mi9_restore',restore_mi9=True)]
 for dose in [.5,1.,1.5]:cases.append(dict(name=f'joint_{dose:g}',negative_sources=['Mi1','Tm3'],restore_mi9=True,dose=dose))
 for theta in [-55.,-60.,-65.]:
  for mode in ['native','frozen','restored','joint']:
   c=dict(name=f'Mi9_theta{int(theta)}_{mode}',mi9_threshold=theta)
   if mode=='frozen':c['freeze_sources']=['Mi9']
   if mode in ['restored','joint']:c['restore_mi9']=True
   if mode=='joint':c['negative_sources']=['Mi1','Tm3']
   cases.append(c)
 return cases


def main():
 (OUT/'curves').mkdir(exist_ok=True);cases=cases_primary();results_path=OUT/'intervention_results.json';records=json.loads(results_path.read_text()) if results_path.exists() else [];completed={r['case']['name'] for r in records};start=time.perf_counter();reference=None
 for case in cases:
  if case['name'] in completed and case['name']!='saved_baseline':
   print('REUSE',case['name'],flush=True);continue
  print('START',case,flush=True);v,p,g=conditional(case)
  if case['name']=='saved_baseline':
   reference=v[:, :,TARGET].clone();original,_=load_data('HH_saved');error=float((v[:,:,TARGET]-original[:,:,TARGET]).abs().max());rms=float((v[:,:,TARGET]-original[:,:,TARGET]).square().mean().sqrt());assert error<.02,(error,rms)
   (OUT/'conditional_equivalence.json').write_text(json.dumps(dict(max_abs_error_mV=error,rmse_mV=rms,outgoing_T4_conductance_zero=True,method='Original HH step, exact precomputed upstream trajectories, unchanged incoming edge order, full initial-state solve.'),indent=2)+'\n')
  if case['name'] in ['saved_repeat','zero_dose']:assert torch.equal(v[:,:,TARGET],reference),case['name']
  metrics=evaluate(case,v,p,g)
  if case['name'] in completed:
   old=next(r for r in records if r['case']['name']==case['name']);assert metrics['cost']==old['cost'] and metrics['dsi_raw']==old['dsi_raw']
  else:records.append(metrics)
  (OUT/'intervention_results.json').write_text(json.dumps(records,indent=2)+'\n')
  print('DONE',case['name'],'cost',metrics['cost'],'DSI',metrics['dsi_raw'],'seconds',time.perf_counter()-start,flush=True);del v,p,g;gc.collect()
 print('INTERVENTIONS_FINISHED',flush=True)
if __name__=='__main__':main()
