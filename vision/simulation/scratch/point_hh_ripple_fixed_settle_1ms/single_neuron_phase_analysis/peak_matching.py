from pathlib import Path
import json,csv
import numpy as np
from scipy.signal import find_peaks
OUT=Path(__file__).resolve().parent;root=OUT.parent/'latest_34401154';q=np.load(root/'comparison_traces.npz');t=q['time_ms'];summary=json.loads((OUT/'analysis_summary.json').read_text());rows=[];stats={}
for cell in ['L1','Mi1','Tm3']:
 j=q['representative_labels'].tolist().index(cell+'_largest_original_oscillation');node=int(q['representative_nodes'][j]);r=next(x['metrics'] for x in summary['latest_34401154']['representatives'] if x['metrics']['node']==node);lag=r['best_lag_ms'];a=q['representative_before'][:,j];b=q['representative_after'][:,j]
 ap,_=find_peaks(a,prominence=max(.05,.2*float(np.ptp(a))),distance=5);bp,_=find_peaks(b,prominence=max(.05,.2*float(np.ptp(b))),distance=5);at=t[ap];bt=t[bp];own=[]
 for ai,ta in zip(ap,at):
  if not 200<=ta<=1150:continue
  bi=int(np.argmin(abs(bt-(ta-lag))));tb=bt[bi]
  row=dict(cell=cell,node=node,A_peak_ms=int(ta),B_matching_peak_ms=int(tb),B_minus_A_peak_time_ms=int(tb-ta),A_peak_mV=float(a[ai]),B_peak_mV=float(b[bp[bi]]));rows.append(row);own.append(row)
 delays=[v['B_minus_A_peak_time_ms'] for v in own];stats[cell]=dict(matched_peak_delay_B_minus_A_ms_min=min(delays),max=max(delays),mean=float(np.mean(delays)),std=float(np.std(delays)),n_pairs=len(delays))
with (OUT/'matched_peak_times.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'matched_peak_summary.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
