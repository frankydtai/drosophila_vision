"""Search existing membrane conductances at fixed dt=1 ms and k_minus=.01."""
from pathlib import Path
from itertools import product
import json,time,gc
import torch
import search as s
OUT=Path(__file__).resolve().parent;original=s.make
mask=torch.isin(s.NC,torch.tensor([s.CELLS.index(c) for c in ['L1','Mi1','Tm3']]))
def make(case):
 p=original(case)
 for param in ['gNa','gK','gLeak']:p[param][mask]*=case.get(param+'_gain',1.)
 return p
s.make=make
cases=[]
for i,(na,kk,leak,fg) in enumerate(product([.3,1.,3.],[.3,1.,3.],[.2,1.],[1.,3.])):
 cases.append(dict(name=f'membrane_{i:03}',forward_gain=fg,feedback_gain=1.,threshold='saved',gNa_gain=na,gK_gain=kk,gLeak_gain=leak))
results=[];start=time.perf_counter()
for i in range(0,len(cases),6):
 print('START',i,'/',len(cases),flush=True);rr=s.run_batch(cases[i:i+6]);results.extend(rr)
 (OUT/'membrane_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
print('FINISHED',flush=True)
