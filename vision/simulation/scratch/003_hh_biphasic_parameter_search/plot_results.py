from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;VAL=OUT/'validation';plt.rcParams.update({'font.size':10})
base=np.load(VAL/'reference_standard.npz');tuned=np.load(VAL/'gain6_standard.npz');hp=np.load(VAL/'HPLP_standard.npz')
cells=list(base['cells']);t=base['time_ms'];on=(t>=0)&(t<160);off=(t>=160)&(t<=1000)
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,cell in zip(axes.flat,['Mi1','Tm3','T4a','L1']):
 ci=cells.index(cell)
 for q,color,label in [(base,'#3675b9','Before: k_minus=0.01'),(tuned,'#d13e37','Tuned: only L1 -> Mi1/Tm3 release')]:ax.plot(t,q['evoked'][:,ci],color=color,lw=1.7,label=label)
 y=tuned['evoked'][:,ci];positive=float(y[on].max());negative=float(y[off].min());ax.set_title(f'{cell} | tuned +{positive:.1f} / {negative:.1f} mV')
 ax.axvspan(0,160,color='#edcf6d',alpha=.2);ax.axhline(0,color='.35',lw=.8);ax.axvline(160,color='.5',lw=.7,ls=':');ax.grid(alpha=.15);ax.set_xlim(-100,1000);ax.set_ylabel('Light - matched background (mV)')
axes[0,0].legend(frameon=False,fontsize=8)
for ax in axes[-1]:ax.set_xlabel('Time from light onset (ms)')
fig.suptitle('Complete 5,789-neuron network | dt = 1 ms | k_minus = 0.01 / ms\n160-ms bright pulse; native HH gates; only two inhibitory pathways tuned',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'biphasic_before_after_1ms.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,cell in zip(axes,['Mi1','Tm3','T4a']):
 for q,color,label in [(base,'#3675b9','HH before'),(tuned,'#d13e37','HH tuned'),(hp,'.35','HPLP 34381872')]:
  ci=list(q['cells']).index(cell);ax.plot(q['time_ms'],q['evoked'][:,ci],color=color,label=label,lw=1.5,ls='--' if label.startswith('HPLP') else '-')
 ax.set_title(cell);ax.axvspan(0,160,color='#edcf6d',alpha=.2);ax.axhline(0,color='.5',lw=.6);ax.grid(alpha=.15);ax.set_ylabel('Evoked Vm (mV)');ax.set_xlim(-100,1000)
axes[0].legend(frameon=False);axes[-1].set_xlabel('Time from light onset (ms)');fig.suptitle('Identical 1-ms pulse test: HH tuning and HPLP reference');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'biphasic_with_hplp_reference.png',dpi=150);plt.close(fig)
if (VAL/'gain6_mbar.npz').exists():
 b=np.load(VAL/'reference_mbar.npz');q=np.load(VAL/'gain6_mbar.npz');fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True,sharey='row')
 for i,cell in enumerate(['Mi1','Tm3','T4a']):
  for j,token in enumerate(['right_bright_w1','left_bright_w1']):
   ax=axes[i,j];tt=q['time__'+token]
   for data,color,label in [(b,'#3675b9','HH before'),(q,'#d13e37','HH tuned')]:ax.plot(tt,data[cell+'__'+token],color=color,lw=1.5,label=label)
   ax.axhline(0,color='.5',lw=.6);ax.axvline(0,color='.5',ls=':',lw=.7);ax.grid(alpha=.15);ax.set_xlim(-200,650);ax.set_title(cell+' '+token.replace('_',' '));ax.set_ylabel('Baseline-subtracted Vm (mV)')
 axes[0,0].legend(frameon=False)
 for ax in axes[-1]:ax.set_xlabel('Time from local first stimulus (ms)')
 fig.suptitle('Moving-bar validation | original full-view geometry | dt=1 ms, k_minus=0.01');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'moving_bar_before_after_1ms.png',dpi=150)
print('PLOTS_FINISHED',flush=True)
