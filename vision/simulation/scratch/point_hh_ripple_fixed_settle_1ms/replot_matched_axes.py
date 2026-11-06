"""Replot saved data only: before/after paired by identical statistic and axes."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
OUT=Path(__file__).resolve().parent
axis_records=[]
for name in ['latest_34401154','original_34400581']:
 folder=OUT/name;path=folder/'comparison_traces.npz';digest=hashlib.sha256(path.read_bytes()).hexdigest();q=np.load(path);cells=q['cells'].tolist();labels=q['representative_labels'].tolist();t=q['time_ms'];mask=(t>=200)&(t<=800)
 for kind in ['population','single_neuron']:
  fig,axes=plt.subplots(3,2,figsize=(11,7.4),sharex=True,sharey='row',layout='constrained')
  for row,cell in enumerate(['L1','Mi1','Tm3']):
   if kind=='population':
    col=cells.index(cell);before=q['population_before'][:,col];after=q['population_after'][:,col];label=cell+' population mean'
   else:
    col=labels.index(cell+'_largest_original_oscillation');before=q['representative_before'][:,col];after=q['representative_after'][:,col];label=cell+f' neuron #{int(q["representative_nodes"][col])}'
   joined=np.r_[before[mask],after[mask]];lo=float(joined.min());hi=float(joined.max());pad=max((hi-lo)*.08,.02);limits=(lo-pad,hi+pad)
   for side,(values,color) in enumerate([(before,'#2874b6'),(after,'#d24b36')]):
    ax=axes[row,side];ax.plot(t[mask],values[mask],color=color,lw=1.05);ax.set_xlim(200,800);ax.set_ylim(*limits);ax.yaxis.set_major_locator(MaxNLocator(nbins=5));ax.tick_params(labelleft=True);ax.grid(alpha=.2);ax.set_ylabel(label+'\nV (mV)')
    if row==0:ax.set_title('Before: original initialization' if side==0 else 'After: +8000 ms segmented settling',color=color,fontsize=11)
    if row==2:ax.set_xlabel('Observation time (ms)')
   assert axes[row,0].get_xlim()==axes[row,1].get_xlim()
   assert axes[row,0].get_ylim()==axes[row,1].get_ylim()
   assert np.array_equal(axes[row,0].get_yticks(),axes[row,1].get_yticks())
   axis_records.append(dict(run=name,figure=kind,cell=cell,before_xlim=list(axes[row,0].get_xlim()),after_xlim=list(axes[row,1].get_xlim()),before_ylim=list(axes[row,0].get_ylim()),after_ylim=list(axes[row,1].get_ylim()),y_ticks=axes[row,0].get_yticks().tolist()))
  fig.suptitle(f'{name} | dt = 1 ms | constant background\nSame x/y limits and ticks in each before/after pair; raw voltages, no smoothing',fontsize=12)
  fig.savefig(folder/f'{kind}_before_after_matched_axes.png',dpi=150);plt.close(fig)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
(OUT/'matched_axes_checks.json').write_text(json.dumps(axis_records,indent=2)+'\n')
print('Verified identical x/y limits and ticks for all 12 before/after pairs; original data unchanged.')
