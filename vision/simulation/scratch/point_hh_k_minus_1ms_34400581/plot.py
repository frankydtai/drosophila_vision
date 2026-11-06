from pathlib import Path
import sys,json,csv,time,hashlib,gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SIM=Path('/home/yitai/drosophila_vision/vision/simulation');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron.forward import forward_v
HH=next((SIM/'0_runs/point_hh').glob('34400581-*'));HP=next((SIM/'0_runs/hp_lp').glob('34381872-*'))
torch.set_num_threads(1);curves={};rows=[]
sources=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/4_forward.py',SIM/'conf/config.yaml',HH/'data/best_param.npz',HP/'data/best_param.npz']
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for model,run in [('HH',HH),('HP-LP',HP)]:
 opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
 for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
 opts['spread_sti_opts']['ms_response']=1200.
 session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);pack=session.packs['spread']['bright'];params=train.params_from_z(z,session);nc=session.connectome.node_cells;cells=list(session.connectome.cells)
 assert session.delta_ms==1. and session.delta_ms_pre==1.
 t=np.arange(pack.i_sti.shape[1])-pack.t_onset
 for k in ([.1,.02,.01] if model=='HH' else [None]):
  label=f'HH k={k:g}' if k is not None else 'HP-LP';print('START',label,flush=True)
  pp={key:(val.clone() if torch.is_tensor(val) else val) for key,val in params.items()}
  if k is not None:pp['k_minus_syn'][:]=k;assert bool((pp['k_minus_syn']==k).all())
  with torch.no_grad():
   v=forward_v(session,pp,pack.i_sti,pack=pack)
   control=forward_v(session,pp,pack.i_sti[:,:1,:].expand_as(pack.i_sti),pack=pack)
  assert torch.isfinite(v).all() and torch.isfinite(control).all()
  cc={'time_ms':t}
  for cell in cells:
   hit=nc==cells.index(cell);pulse=v[0,:,hit].mean(-1).numpy();ctl=control[0,:,hit].mean(-1).numpy();y=pulse-ctl;cc[cell]=y;cc[cell+'_pulse']=pulse;cc[cell+'_control']=ctl
   on=(t>=0)&(t<160);off=(t>=160)&(t<=1100);pol=1 if y[on].max()>=-y[on].min() else -1;oo=np.maximum(-pol*y[off],0);pk=int(oo.argmax());amp=float(oo[pk]);above=oo>=amp/2;lo=hi=pk
   while lo>0 and above[lo-1]:lo-=1
   while hi+1<len(above) and above[hi+1]:hi+=1
   rows.append(dict(model=label,cell=cell,on_min_mV=float(y[on].min()),on_max_mV=float(y[on].max()),opposite_off_peak_mV=amp,opposite_off_fwhm_ms=float(hi-lo+1) if amp>0 else 0.,opposite_off_delay_ms=float(t[off][pk]-160)))
  curves[label]=cc;np.savez_compressed(OUT/('hp_lp.npz' if k is None else f'hh_k_{k:g}.npz'),**cc)
  print('DONE',label,flush=True);del v,control;gc.collect()
with (OUT/'metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Same physical drive: a 160 ms bright pulse, original network and trained parameters.
colors={'HH k=0.1':'#2765b0','HH k=0.02':'#e28c1c','HH k=0.01':'#cf3838','HP-LP':'#222222'}
labels={'HH k=0.1':'HH k_minus=0.1 (original)','HH k=0.02':'HH k_minus=0.02','HH k=0.01':'HH k_minus=0.01','HP-LP':'HP-LP reference'}
fig,axes=plt.subplots(3,2,figsize=(12,8.8))
for row,cell in enumerate(['L1','Mi1','Tm3']):
 for col in range(2):
  ax=axes[row,col]
  for label,cc in curves.items():
   if col==1 and label=='HP-LP':continue
   ax.plot(cc['time_ms']-(160 if col else 0),cc[cell],color=colors[label],lw=1.6 if label!='HP-LP' else 1.8,ls='--' if label=='HP-LP' else '-',label=labels[label])
  ax.axhline(0,color='0.5',lw=.7);ax.grid(alpha=.15);ax.set_ylabel(cell+' response (mV)')
  if col==0:
   ax.axvspan(0,160,color='#f3d75b',alpha=.2);ax.axvline(160,color='0.5',ls=':',lw=.8);ax.set_xlim(-60,1000)
   if row==0:ax.set_title('Light ON / OFF: HH and HP-LP')
   if row==2:ax.set_xlabel('Time from light onset (ms)')
  else:
   ax.axvline(0,color='0.5',ls=':',lw=.8);ax.set_xlim(0,750)
   # Zoom to the opposite-polarity OFF lobe, retaining physical mV units.
   vals=np.concatenate([cc[cell][(cc['time_ms']>=180)&(cc['time_ms']<=900)] for label,cc in curves.items() if label!='HP-LP'])
   if cell=='L1':ax.set_ylim(-.7,max(float(vals.max())*1.15,1.))
   else:ax.set_ylim(min(float(vals.min())*1.15,-.7),.7)
   if row==0:ax.set_title('HH only: opposite peak after light OFF (zoom)')
   if row==2:ax.set_xlabel('Time from light offset (ms)')
axes[0,0].legend(frameon=False,fontsize=8,loc='lower right');axes[0,1].legend(frameon=False,fontsize=8,loc='upper right')
fig.suptitle('Fixed delta_ms = delta_ms_pre = 1 ms | light ON 0-160 ms\nPopulation means, paired constant-light control subtracted; HH changes only k_minus_syn',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'k_minus_comparison_1ms.png',dpi=160);plt.close(fig)
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources};assert before==after,'source changed during evaluation'
(OUT/'manifest.json').write_text(json.dumps(dict(delta_ms=1.,delta_ms_pre=1.,light_on_ms=0,light_off_ms=160,hh_run=str(HH),hp_lp_run=str(HP),source_sha256=before,source_unchanged=True,hh_k_minus_values=[.1,.02,.01],metrics=rows),indent=2)+'\n')
print('IMAGE',OUT/'k_minus_comparison_1ms.png',flush=True)
for r in rows:
 if r['cell'] in ['L1','Mi1','Tm3']:print(json.dumps(r),flush=True)
