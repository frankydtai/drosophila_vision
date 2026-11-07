"""Preserve tonic E/I balance while increasing existing feedback gains."""
from pathlib import Path
from itertools import product
import json,time,gc
import numpy as np
import torch
import search as s
OUT=Path(__file__).resolve().parent
with torch.no_grad():
 p=s.hh.prepare_params(s.BASE,s.I0,s.SESSION);state=s.hh.pre_steady(s.SESSION,p,1,i_sti=s.I0.unsqueeze(1));vs=[]
 for i in range(1400):
  if i>=1200:vs.append(state['v'][0].clone())
  state=s.hh.step(state,p,s.I0,s.SESSION,delta_ms=1.)
 vref=torch.stack(vs).mean(0)
 premean=torch.zeros(len(s.CELLS));premean.scatter_add_(0,s.NC,vref);premean/=s.COUNTS
original=s.make
R1=s.CELLS.index('R1-6');L1=s.CELLS.index('L1')
BALANCE={L1:(s.EDGE_PRE==R1)&(s.EDGE_POST==L1)}
for cell in ['Mi1','Tm3']:
 ci=s.CELLS.index(cell);BALANCE[ci]=torch.isin(s.EDGE_PRE,torch.tensor([s.CELLS.index(c) for c in ['L3','L5']]))&(s.EDGE_POST==ci)&(s.BASE['e_syn']==0)
vin=vref[s.SRC];post=vref[s.TAR]
def edge_current(p):
 return p['gS']*torch.sigmoid((vin-p['v_th_syn'])/p['delta_syn'])*(p['e_syn']-post)
basecurrent=edge_current(s.BASE)
param_details={}
def make_balanced(case):
 # Restrict the release-threshold change to the reciprocal feedback motif.
 p={k:s.BASE[k].clone() for k in s.COLUMNS}
 p['gS'][s.FORWARD]*=case.get('forward_gain',1.)
 p['gS'][s.FEEDBACK]*=case.get('feedback_gain',1.)
 if case.get('threshold','saved')=='rest':
  mask=s.FORWARD|s.FEEDBACK;p['v_th_syn'][mask]=premean[s.EDGE_PRE[mask]]
 delta=edge_current(p)-basecurrent;balance_gains={}
 for ci,edges in BALANCE.items():
  dc=float(delta[s.EDGE_POST==ci].sum());denom=float(basecurrent[edges].sum());factor=1-dc/denom
  if factor<=0:raise ValueError((case,s.CELLS[ci],factor))
  p['gS'][edges]*=factor;balance_gains[s.CELLS[ci]]=factor
 residual=edge_current(p)-basecurrent
 param_details[case['name']]=dict(case=case,balance_pathway_gains=balance_gains,target_cell_mean_net_current_change_pA={s.CELLS[ci]:float(residual[s.EDGE_POST==ci].sum()/s.COUNTS[ci]) for ci in BALANCE})
 return p
s.make=make_balanced
cases=[]
for i,(fg,bg,th) in enumerate(product([1.,2.,4.],[1.,3.,10.,30.],['saved','rest'])):
 c=dict(name=f'balanced_{i:03}',forward_gain=fg,feedback_gain=bg,threshold=th)
 try:make_balanced(c);cases.append(c)
 except ValueError as e:print('SKIP',e,flush=True)
results=[];start=time.perf_counter()
for i in range(0,len(cases),6):
 print('START',i,'/',len(cases),flush=True);rows=s.run_batch(cases[i:i+6]);results.extend(rows)
 (OUT/'balanced_results.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'balanced_parameter_details.json').write_text(json.dumps(param_details,indent=2)+'\n')
 print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
print('FINISHED',flush=True)
