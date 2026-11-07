"""Dose/timing tests and background-normalized effective-output restoration."""
from pathlib import Path
import json,time,gc
import numpy as np
import torch
import interventions as ex
OUT=Path(__file__).resolve().parent
P,z,p=ex.ref.load(ex.ref.HP);HP_STIM=torch.from_numpy(np.load(OUT/'HPLP_matched_full.npy'));HP_BG=torch.from_numpy(np.load(OUT/'HPLP_matched_background.npy'));theta=p['v_th'].reshape(1,1,-1)
bg=torch.relu(HP_BG-theta);stim=torch.relu(HP_STIM-theta);valid=bg>1.;RATIO=torch.where(valid,stim/bg.clamp_min(1.),torch.ones_like(bg)).clamp(0.,2.)
coverage={c:float(valid[:,:,ex.NC==ex.CELLS.index(c)].float().mean()) for c in ['Mi9','Mi1','Tm3','L1','L2']};(OUT/'output_ratio_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n');del P,z,p,HP_STIM,HP_BG,bg,stim,valid;gc.collect()
ORIGINAL_HOOK=ex.hook;BG_CACHE={}
def hook(case,clock,hh_delta,edge_ids):
 if not case.get('probability_sources'):return ORIGINAL_HOOK(case,clock,hh_delta,edge_ids)
 label=case.get('k_label','HH_saved')
 if label not in BG_CACHE:
  BG_CACHE.clear();BG_CACHE[label]=torch.from_numpy(np.load(OUT/(label+'_background.npy')))
 background=BG_CACHE[label];es=ex.SRC[edge_ids];precell=ex.NC[es]
 masks={c:precell==ex.CELLS.index(c) for c in case['probability_sources']}
 def release(s,pre,threshold,delta,kminus,dt):
  step=clock['t'];dose=case.get('dose',1.)
  if step>=ex.ONSET and dose!=0:
   j=step-1-int(case.get('shift_ms',0));ratio=RATIO[:,min(max(j,0),ex.NT-1),es]
   if j<ex.ONSET:ratio=torch.ones_like(ratio)
   if case.get('swap_directions'):ratio=ratio.flip(0)
   bg_inf=ex.ref.hh.release_steady(background[:,step-1,es],threshold,delta);native_inf=ex.ref.hh.release_steady(pre,threshold,delta);modified=native_inf.clone()
   for source,mask in masks.items():
    if source=='Mi9':
     rr=ratio[:,mask] if case.get('mi9_full_ratio',True) else ratio[:,mask].clamp_max(1.)
     modified[:,mask]=(1-dose)*native_inf[:,mask]+dose*bg_inf[:,mask]*rr
    else:
     change=native_inf[:,mask]-bg_inf[:,mask];desired=dose*bg_inf[:,mask]*(ratio[:,mask].clamp_max(1.)-1.)
     modified[:,mask]=bg_inf[:,mask]+change.clamp_min(0)+torch.minimum(change.clamp_max(0),desired)
   active=torch.stack(list(masks.values())).any(0);safe=modified[:,active].clamp(1e-7,1-1e-7);pre=pre.clone();pre[:,active]=threshold[active]+delta[active]*torch.logit(safe)
  return ex.OLD_RELEASE(s,pre,threshold,delta,kminus,dt)
 return release
ex.hook=hook
# Extend conditional's mode switch without changing its HH integration.
original_conditional=ex.conditional
def conditional(case):
 cc=dict(case)
 if cc.get('probability_sources'):cc['negative_sources']=[];cc['restore_mi9']=True
 return original_conditional(cc)

cases=[]
for dose in [.02,.05,.1,.2,.3]:cases.append(dict(name=f'fine_negative_{dose:g}',negative_sources=['Mi1','Tm3'],dose=dose))
for shift in [-150,-100,-50,50,100]:cases.append(dict(name=f'phase_negative_{shift:+d}',negative_sources=['Mi1','Tm3'],dose=.2,shift_ms=shift))
for dose in [.25,.5,1.]:cases.append(dict(name=f'ratio_exc_{dose:g}',probability_sources=['Mi1','Tm3'],dose=dose))
for th in [None,-55.,-60.,-65.]:
 for full in [False,True]:cases.append(dict(name=f'ratio_Mi9_{th}_{full}',probability_sources=['Mi9'],mi9_threshold=th,mi9_full_ratio=full,dose=1.))
for th in [None,-55.,-60.]:
 for dose in [.25,.5]:cases.append(dict(name=f'ratio_joint_{th}_{dose:g}',probability_sources=['Mi9','Mi1','Tm3'],mi9_threshold=th,mi9_full_ratio=True,dose=dose))
results=[];start=time.perf_counter()
for case in cases:
 print('START',case,flush=True);v,p,g=conditional(case);r=ex.evaluate(case,v,p,g);results.append(r);(OUT/'followup_results.json').write_text(json.dumps(results,indent=2)+'\n')
 print('DONE',case['name'],round(r['cost']),r['dsi_raw'],'seconds',time.perf_counter()-start,flush=True);del v,p,g;gc.collect()
print('FOLLOWUPS_FINISHED',flush=True)
