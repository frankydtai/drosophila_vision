from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1]
q=np.load(SIM/'scratch/point_hh_ripple_fixed_settle_1ms/latest_34401154/extreme20_original_1ms/selected_traces.npz');axis=json.loads((OUT/'manifest.json').read_text())['same_global_axes'];t=q['time_ms'];window=(t>=200)&(t<=1200)
fig,axes=plt.subplots(5,4,figsize=(16,11),sharex=True,sharey=True,layout='constrained')
for group,base,colbase,color in [('largest',0,0,'#cf4438'),('smallest',10,2,'#2875b4')]:
 for k in range(10):
  i=base+k;ax=axes[k%5,colbase+k//5];y=q['voltage_mV'][:,i];sd=float(q['std_mV'][i]);label=f'{sd:.3f}' if sd>=.001 else f'{sd:.2g}'
  ax.plot(t[window],y[window],color=color,lw=.85);ax.set_xlim(*axis['xlim']);ax.set_ylim(*axis['ylim']);ax.set_yticks(axis['yticks']);ax.tick_params(labelleft=True,labelsize=8);ax.grid(alpha=.17);ax.set_title(f'#{k+1} {q["cell_types"][i]} | node {q["nodes"][i]}\nSD {label} mV; range {np.ptp(y[window]):.3f} mV',fontsize=9,color=color)
  if k%5==4:ax.set_xlabel('Observation time (ms)',fontsize=9)
  if colbase+k//5 in [0,2]:ax.set_ylabel('Raw V (mV)',fontsize=9)
for ax in axes.flat:assert ax.get_ylim()==tuple(axis['ylim'])
fig.suptitle('10 largest fluctuations (left two columns)                         10 smallest fluctuations (right two columns)\nPoint HH 34401154 | dt=1 ms | constant background 20 pA at photoreceptors | original initialization\nSame saved HH neurons and traces, re-rendered on the exact HPLP axis limits',fontsize=12)
fig.savefig(OUT/'hh_reference_identical_axes.png',dpi=150)
