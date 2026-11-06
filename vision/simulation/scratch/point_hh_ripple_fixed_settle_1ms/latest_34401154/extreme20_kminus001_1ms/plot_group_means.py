from pathlib import Path
import hashlib,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;source=OUT/'selected_traces.npz';digest=hashlib.sha256(source.read_bytes()).hexdigest();q=np.load(source);manifest=json.loads((OUT/'manifest.json').read_text())
assert manifest['k_minus_syn_per_ms']==.01 and manifest['dt_ms']==1.
v=q['voltage_mV'].astype(np.float64);t=q['time_ms'];assert v.shape==(len(t),20)
large=v[:,:10].mean(axis=1);small=v[:,10:].mean(axis=1);mask=(t>=200)&(t<=1200)
np.savez_compressed(OUT/'group_mean_traces.npz',time_ms=t,largest10_mean_mV=large,smallest10_mean_mV=small,largest_nodes=q['nodes'][:10],smallest_nodes=q['nodes'][10:])
with (OUT/'group_mean_traces.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['time_ms','largest10_mean_mV','smallest10_mean_mV']);w.writerows(zip(t,large,small))
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');ax.plot(t[mask],large[mask],color='#cf4438',lw=2,label='Mean of 10 largest-fluctuation neurons');ax.plot(t[mask],small[mask],color='#2875b4',lw=2,label='Mean of 10 smallest-fluctuation neurons');ax.set_xlim(200,1200);ax.set_xlabel('Observation time (ms)');ax.set_ylabel('Mean raw membrane voltage (mV)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9);ax.set_title('k_minus_syn = 0.01 /ms | dt = 1 ms | constant background\nArithmetic mean of the same 10 neurons in each group; no smoothing',fontsize=12);fig.savefig(OUT/'largest10_smallest10_group_means.png',dpi=170);plt.close(fig)
assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
print('Verified two arithmetic means, each over exactly 10 previously selected neurons; source unchanged.')
