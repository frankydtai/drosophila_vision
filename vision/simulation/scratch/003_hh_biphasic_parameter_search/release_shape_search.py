"""Only existing shared release threshold/slope; native HH kinetics."""
from pathlib import Path
from itertools import product
import json,time,gc
import search as s
OUT=Path(__file__).resolve().parent;original=s.make

def make(case):
 p=original(case);p['delta_syn'][:]=case['delta_syn_mV'];p['v_th_syn'][:]=case['v_th_syn_mV'];return p
s.make=make
cases=[]
for i,(delta,threshold) in enumerate(product([.5,1.,2.,4.],[-55.,-50.,-45.,-40.,-35.])):
 cases.append(dict(name=f'release_shape_{i:03}',delta_syn_mV=delta,v_th_syn_mV=threshold))
results=[];start=time.perf_counter()
for i in range(0,len(cases),5):
 print('START',i,'/',len(cases),flush=True);rr=s.run_batch(cases[i:i+5]);results.extend(rr)
 (OUT/'release_shape_results.json').write_text(json.dumps(results,indent=2)+'\n');print('DONE',i,'elapsed',time.perf_counter()-start,'top',[(r['case'],round(r['score'],3)) for r in sorted(results,key=lambda r:r['score'],reverse=True)[:3]],flush=True);gc.collect()
print('FINISHED',flush=True)
