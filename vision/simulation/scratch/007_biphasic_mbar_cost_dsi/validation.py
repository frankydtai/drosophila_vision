"""Timing sensitivity, secondary k control, and original full-network HH validation."""
from pathlib import Path
import json,gc,time
import numpy as np
import torch
import interventions as ex
OUT=Path(__file__).resolve().parent
OLD_STEP=ex.ref.hh.step
ALL=torch.arange(ex.SRC.numel())
results=[]
def save(r):
 results.append(r);(OUT/'validation_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',r['case']['name'],r['cost'],r['dsi_raw'],flush=True)

def full(case):
 label=case.get('k_label','HH_saved');vref,dd=ex.load_data(label);p=ex.make(case);clock={'t':0};background=torch.from_numpy(np.load(OUT/(label+'_background.npy')))
 reb=torch.isin(ex.NC[ex.SRC],torch.tensor([ex.CELLS.index(c) for c in case.get('rebound_sources',[])]));chosen=ex.hook(case,clock,dd,ex.EI)
 g=torch.zeros(2,ex.NT,int(ex.TARGET.sum()));mi9=ex.EPRE==ex.CELLS.index('Mi9')
 def release(s,pre,th,d,k,dt):
  ti=clock['t'];effective=pre
  if ti>=ex.ONSET and bool(reb.any()):
   effective=pre.clone();base=background[:,ti-1,ex.SRC[reb]];change=pre[:,reb]-base;hp=ex.HP_DELTA[:,ti-1,ex.SRC[reb]].clamp_min(0);dose=case.get('dose',1.)
   effective[:,reb]=base+change.clamp_max(0)+(1-dose)*change.clamp_min(0)+dose*hp
  out=ex.OLD_RELEASE(s,effective,th,d,k,dt)
  if case.get('negative_sources') or case.get('restore_mi9'):
   out[:,ex.EI]=chosen(s[:,ex.EI],effective[:,ex.EI],th[ex.EI],d[ex.EI],k[ex.EI],dt)
  return out
 def step(state,params,i_sti,session,**kwargs):
  clock['t']+=1
  if clock['t']==1:
   gg=torch.zeros_like(state['v']);gg.scatter_add_(1,ex.ETAR[mi9].expand(2,-1),state['s'][:,ex.EI][:,mi9]*p['gS'][ex.EI][mi9]);g[:,0]=gg[:,ex.TARGET]
  out=OLD_STEP(state,params,i_sti,session,**kwargs);gg=torch.zeros_like(out['v']);gg.scatter_add_(1,ex.ETAR[mi9].expand(2,-1),out['s'][:,ex.EI][:,mi9]*p['gS'][ex.EI][mi9]);g[:,clock['t']]=gg[:,ex.TARGET];return out
 ex.ref.hh._release_step=release;ex.ref.hh.step=step
 try:
  with torch.no_grad():v=ex.ref.forward_pack(ex.H,p,ex.PACK.i_sti,ex.PACK)
 finally:ex.ref.hh._release_step=ex.OLD_RELEASE;ex.ref.hh.step=OLD_STEP
 assert clock['t']==ex.NT-1 and torch.isfinite(v).all();r=ex.evaluate(case,v,p,g)
 if case.get('rebound_sources')==['L2']:r['max_abs_difference_from_saved_mV']=float((v-vref).abs().max())
 if case.get('check_conditional'):
  vc,pc,gc_=ex.conditional(case);r['conditional_max_abs_error_mV']=float((vc-v).abs().max());assert r['conditional_max_abs_error_mV']<.02
 del v,p,g,background;gc.collect();return r

if __name__=='__main__':
 for shift in [-75,-50,-25]:
  for dose in [.1,.15,.2,.25]:
   case=dict(name=f'grid_neg_d{dose:g}_t{shift}',negative_sources=['Mi1','Tm3'],dose=dose,shift_ms=shift);v,p,g=ex.conditional(case);save(ex.evaluate(case,v,p,g));del v,p,g;gc.collect()
 for case in [dict(name='k001_baseline',k_label='HH_k001'),dict(name='k001_small_negative',k_label='HH_k001',negative_sources=['Mi1','Tm3'],dose=.05),dict(name='k001_negative_1',k_label='HH_k001',negative_sources=['Mi1','Tm3'],dose=1.)]:
  v,p,g=ex.conditional(case);save(ex.evaluate(case,v,p,g));del v,p,g;gc.collect()
 edge_proof={c:dict(n_edges=int((ex.NC[ex.SRC]==ex.CELLS.index(c)).sum()),nonzero_edges=int(((ex.NC[ex.SRC]==ex.CELLS.index(c))&(ex.BASE['gS']!=0)).sum())) for c in ['L1','L2']};(OUT/'rebound_edge_proof.json').write_text(json.dumps(edge_proof,indent=2)+'\n')
 for case in [dict(name='full_small_negative',negative_sources=['Mi1','Tm3'],dose=.05,check_conditional=True),dict(name='full_phase_negative',negative_sources=['Mi1','Tm3'],dose=.2,shift_ms=-50,check_conditional=True),dict(name='full_L1_rebound_half',rebound_sources=['L1'],dose=.5),dict(name='full_L1_rebound',rebound_sources=['L1'],dose=1.),dict(name='full_L2_rebound',rebound_sources=['L2'],dose=1.)]:
  print('FULL_START',case,flush=True);save(full(case))
 print('VALIDATION_FINISHED',flush=True)
