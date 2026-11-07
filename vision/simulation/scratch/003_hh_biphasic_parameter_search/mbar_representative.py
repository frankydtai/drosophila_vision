"""Direct moving-bar search of L3/L5 release operating points, native 1 ms HH."""
from pathlib import Path
from itertools import product
import json,time,gc
import numpy as np
import torch
import search as s
from neuron.forward import _mbar_gaussian_overlay
from train.param import override_val_from
from train.cost import _pack_cost_parts_from_v_readout,_weighted_mean_cost_parts,gt_affine_from_pack,pack_traces
from figure.mbar import _traces_from_forward
from task.mbar.pack import bar_specs_from_task,dsi_values
import task.mbar.pack as mp
OUT=Path(__file__).resolve().parent;original=mp.build_mbar_signals
mp.build_mbar_signals=lambda *a,**kw:original(*a,**dict(kw,use_cache=False))
opts=json.loads((s.RUN/'data/train_opts.json').read_text());opts['device']='cpu';ss=s.train.resolve_session(opts);pack=ss.packs['mbar']['bright'];specs=bar_specs_from_task(ss,'mbar','bright');tokens=[x.token for x in specs]
assert ss.delta_ms==1. and ss.delta_ms_pre==1.

def make(case):
 p={k:s.BASE[k].clone() for k in s.COLUMNS}
 if case.get('tuned_L1',False):p['gS'][s.FORWARD]*=3.;p['v_th_syn'][s.FORWARD]=-51.;p['delta_syn'][s.FORWARD]=.5
 for cell in ['L3','L5']:
  if case.get(cell+'_threshold') is not None:
   hit=s.EDGE_PRE==s.CELLS.index(cell);p['v_th_syn'][hit]=case[cell+'_threshold'];p['delta_syn'][hit]=case.get('source_delta',2.)
 return p
s.make=make
cases=[dict(name='mbar_centered_native',L3_threshold=-65.,L5_threshold=-55.,source_delta=2.,tuned_L1=False),dict(name='mbar_centered_tuned',L3_threshold=-65.,L5_threshold=-55.,source_delta=2.,tuned_L1=True)]
results=[];clock=time.perf_counter();nt=pack.i_sti.shape[1]
with torch.no_grad():
 for start in range(0,len(cases),4):
  batch=cases[start:start+4];b=len(batch)*2;pp=s.batched(batch);i0=pack.i_sti[:,0].repeat(len(batch),1);state=s.hh.pre_steady(ss,pp,b,i_sti=i0.unsqueeze(1));trace=torch.empty(b,nt,s.CON.n_node)
  print('START',start,'/',len(cases),flush=True)
  for ti in range(nt):
   trace[:,ti]=state['v']
   if ti==nt-1:break
   drive=_mbar_gaussian_overlay(pack.i_sti[:,ti],s.BASE,pack,t=ti).repeat(len(batch),1)
   state=s.hh.step(state,pp,drive,ss,delta_ms=1.)
  assert torch.isfinite(trace).all()
  for j,c in enumerate(batch):
   v=trace[2*j:2*j+2];p=dict(s.BASE);p.update(make(c));override_val_from(p,ss,onset_trace=v,t_onset=pack.t_onset)
   parts=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,ss,*gt_affine_from_pack(p,pack,ss),pack_traces(v,pack),trace=v),ss);ds={k:float(x) for k,x in dsi_values(pack,v).items()}
   tr,*_=_traces_from_forward(ss,'mbar','bright',v.numpy(),specs,tokens);curves={};metrics={};scores=[]
   for (cell,token),y in tr.ca_mean_cell.items():
    tt=np.arange(len(y))-tr.before_t[token];base=float(y[(tt>=-350)&(tt<=-180)].mean());yy=y-base;active=(tt>=-150)&(tt<=576);pos=float(yy[active].max());neg=float(yy[active].min());metrics[cell+'__'+token]=dict(positive_peak_mV=pos,negative_peak_mV=neg)
    curves[cell+'__'+token]=yy;curves['time__'+token]=tt
    if cell in ['Mi1','Tm3']:scores.append(min(pos,max(0.,-neg)))
   score=.5*min(scores)+.5*np.mean(scores);record=dict(case=c,score=float(score),cost=sum(float(x) for x in parts.values()),dsi_raw=ds,metrics=metrics)
   results.append(record);np.savez_compressed(OUT/'curves'/f"{c['name']}.npz",**curves)
  (OUT/'mbar_representative_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',start,'seconds',time.perf_counter()-clock,'top',[(r['case'],round(r['score'],3),round(r['cost'])) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True)
  del trace;gc.collect()
print('FINISHED',flush=True)
