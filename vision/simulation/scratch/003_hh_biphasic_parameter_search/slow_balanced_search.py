"""Experimental slow K + calibrated resting leak reversals; no formal edits."""
from pathlib import Path
from itertools import product
import json,time,gc
import torch
import search as s
OUT=Path(__file__).resolve().parent;original_rates=s.hh.gate_rates;original_batch=s.batched;SCALE=None
TARGETS=['L1','L2','Mi1','Tm3'];NMASK=torch.isin(s.NC,torch.tensor([s.CELLS.index(c) for c in TARGETS]));EMASK=NMASK[s.TAR]
with torch.no_grad():
 pp=s.hh.prepare_params(s.BASE,s.I0,s.SESSION);st=s.hh.pre_steady(s.SESSION,pp,1,i_sti=s.I0.unsqueeze(1));vv=[]
 for ti in range(1400):
  if ti>=1200:vv.append(st['v'][0].clone())
  st=s.hh.step(st,pp,s.I0,s.SESSION,delta_ms=1.)
 vref=torch.stack(vv).mean(0);am,bm,ah,bh,an,bn=original_rates(vref);m=am/(am+bm);h=ah/(ah+bh);n=an/(an+bn)

def inward(p):
 currents=p['gNa']*m**3*h*(p['eNa']-vref)+p['gK']*n**4*(p['eK']-vref)+p['gLeak']*(p['eLeak']-vref)
 es=p['gS']*torch.sigmoid((vref[s.SRC]-p['v_th_syn'])/p['delta_syn'])*(p['e_syn']-vref[s.TAR])
 currents=currents.clone();currents.scatter_add_(0,s.TAR,es);return currents
baseline=inward(s.BASE);details={}
def make(case):
 p={k:s.BASE[k].clone() for k in s.COLUMNS};p['gS'][EMASK]*=case['incoming_syn_gain'];p['cap'][NMASK]=case['cap_pF']
 p['gNa'][NMASK]=case.get('gNa_nS',10.);p['gK'][NMASK]=case.get('gK_nS',3.);p['gLeak'][NMASK]=case.get('gLeak_nS',.3)
 delta=inward(p)-baseline;offsets={}
 for c in TARGETS:
  mask=s.NC==s.CELLS.index(c);offset=-delta[mask].sum()/p['gLeak'][mask].sum();p['eLeak'][mask]+=offset
  value=float(p['eLeak'][mask][0]);offsets[c]=value
  if not -110<=value<=40:raise ValueError((case,c,value))
 details[case['name']]=dict(parameters=case,eLeak_mV=offsets)
 return p
s.make=make

def batch(cases):
 global SCALE
 p=original_batch(cases);SCALE=torch.ones(len(cases)*2,s.CON.n_node)
 for j,c in enumerate(cases):SCALE[2*j:2*j+2,NMASK]=c['n_rate']
 return p
s.batched=batch

def rates(v):
 am,bm,ah,bh,an,bn=original_rates(v);return am,bm,ah,bh,an*SCALE,bn*SCALE
s.hh.gate_rates=rates
cases=[]
for i,(rate,gs,cap) in enumerate(product([1.,.1,.02],[.03,.1,.3,1.],[1.,10.,30.])):
 c=dict(name=f'slow_balanced_{i:03}',n_rate=rate,incoming_syn_gain=gs,cap_pF=cap,gNa_nS=10.,gK_nS=3.,gLeak_nS=.3)
 try:make(c);cases.append(c)
 except ValueError as e:print('SKIP',e,flush=True)
results=[];start=time.perf_counter()
try:
 for i in range(0,len(cases),6):
  print('START',i,'/',len(cases),flush=True);rows=s.run_batch(cases[i:i+6]);results.extend(rows)
  (OUT/'slow_balanced_results.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'slow_balanced_parameters.json').write_text(json.dumps(details,indent=2)+'\n')
  print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
finally:s.hh.gate_rates=original_rates
print('FINISHED',flush=True)
