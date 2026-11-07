"""Experimental K recovery-rate override; repository kernel is never edited."""
from pathlib import Path
from itertools import product
import json,time,gc
import torch
import search as s
OUT=Path(__file__).resolve().parent
TARGETS=['L1','L2','Mi1','Tm3'];NMASK=torch.isin(s.NC,torch.tensor([s.CELLS.index(c) for c in TARGETS]));EMASK=NMASK[s.TAR]
original_make=s.make;original_batch=s.batched;original_rates=s.hh.gate_rates;SCALE=None

def make(case):
 p={k:s.BASE[k].clone() for k in s.COLUMNS}
 p['gS'][EMASK]*=case['incoming_syn_gain'];p['cap'][NMASK]=case['cap_pF']
 if case['membrane']=='graded':
  p['gNa'][NMASK]=10.;p['gK'][NMASK]=3.;p['gLeak'][NMASK]=.3
 return p
s.make=make

def batch(cases):
 global SCALE
 p=original_batch(cases);SCALE=torch.ones(len(cases)*2,s.CON.n_node)
 for j,c in enumerate(cases):SCALE[2*j:2*j+2,NMASK]=c['n_rate']
 return p
s.batched=batch

def rates(v):
 am,bm,ah,bh,an,bn=original_rates(v)
 return am,bm,ah,bh,an*SCALE,bn*SCALE
s.hh.gate_rates=rates
cases=[]
for i,(kind,rate,gs,cap) in enumerate(product(['current','graded'],[1.,.1,.02],[.03,.1,.3,1.],[1.,10.])):
 cases.append(dict(name=f'slow_recovery_{i:03}',membrane=kind,n_rate=rate,incoming_syn_gain=gs,cap_pF=cap))
results=[];start=time.perf_counter()
try:
 for i in range(0,len(cases),6):
  print('START',i,'/',len(cases),flush=True);rr=s.run_batch(cases[i:i+6]);results.extend(rr)
  (OUT/'slow_recovery_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
finally:s.hh.gate_rates=original_rates
print('FINISHED',flush=True)
