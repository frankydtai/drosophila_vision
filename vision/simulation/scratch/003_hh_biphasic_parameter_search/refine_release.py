"""Refine the successful native-HH inhibitory-release parameter region."""
from pathlib import Path
from itertools import product
import json,time,gc
import search as s
OUT=Path(__file__).resolve().parent;original=s.make

def make(case):
 p=original(case);p['delta_syn'][s.FORWARD]=case['forward_delta_mV'];p['v_th_syn'][s.FORWARD]=case['forward_threshold_mV'];return p
s.make=make
cases=[]
for i,(fg,th,delta) in enumerate(product([1.,3.,6.],[-52.,-51.,-50.,-49.],[.5,.75,1.])):
 cases.append(dict(name=f'refined_release_{i:03}',forward_gain=fg,feedback_gain=1.,threshold='saved',forward_delta_mV=delta,forward_threshold_mV=th))
results=[];start=time.perf_counter()
for i in range(0,len(cases),6):
 print('START',i,'/',len(cases),flush=True);rr=s.run_batch(cases[i:i+6]);results.extend(rr)
 (OUT/'refined_release_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
print('FINISHED',flush=True)
