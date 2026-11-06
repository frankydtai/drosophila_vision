"""Rank all neurons under the exact saved original-initialization background run."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SIM=Path('/home/yitai/drosophila_vision/vision/simulation');OUT=Path(__file__).resolve().parent;PARENT=OUT.parent
sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron import point_hh
old_summary=json.loads((PARENT/'summary.json').read_text());run=Path(old_summary['run']);opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
torch.set_num_threads(1)
sources=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/4_forward.py',run/'data/best_param.npz',run/'data/train_opts.json',PARENT/'original_initial_state.npz',Path(__file__)]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);params=train.params_from_z(z,session);I=session.packs['spread']['bright'].i_sti[:,0,:];pp=point_hh.prepare_params(params,I,session)
loaded=np.load(PARENT/'original_initial_state.npz');names={'v':'v','m':'HH_m','h':'HH_h','n':'HH_n','s':'Syn_s'};state={k:torch.from_numpy(loaded[name].copy())[None,:] for k,name in names.items()};N=session.connectome.n_node
with torch.no_grad():
 voltage=torch.empty((1201,N),dtype=I.dtype);voltage[0]=state['v'][0]
 for ti in range(1,1201):
  state=point_hh.step(state,pp,I,session,delta_ms=1.);voltage[ti]=state['v'][0]
assert torch.isfinite(voltage).all()
v=voltage.numpy();t=np.arange(1201);window=(t>=200)&(t<=1200);old=np.load(PARENT/'comparison_traces.npz');rep=old['representative_nodes'];reference_error=float(np.max(abs(v[:,rep]-old['representative_before'])));assert reference_error==0.,reference_error
# Center before accumulation, in float64: exactly constant traces must have exactly zero SD.
vv=v[window].astype(np.float64);centered=vv-vv[0];std=np.std(centered,axis=0,dtype=np.float64);ptp=np.ptp(vv,axis=0);assert np.all(std[ptp==0]==0)
idx=np.arange(N);largest=np.lexsort((idx,-std))[:10];smallest=np.lexsort((idx,std))[:10];assert len(set(largest)&set(smallest))==0
cells=list(session.connectome.cells);nc=session.connectome.node_cells.numpy();node_ids=np.asarray(session.connectome.node_ids);us=np.asarray(session.connectome.us);vs=np.asarray(session.connectome.vs)
rows=[]
for group,ns in [('largest',largest),('smallest',smallest)]:
 for rank,node in enumerate(ns,1):rows.append(dict(group=group,rank=rank,node=int(node),root_id=str(node_ids[node]),cell_type=cells[nc[node]],u=int(us[node]),v=int(vs[node]),std_mV=float(std[node]),peak_to_peak_mV=float(ptp[node]),mean_mV=float(vv[:,node].mean())))
with (OUT/'selected_20_neurons.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
allrows=[dict(node=int(i),root_id=str(node_ids[i]),cell_type=cells[nc[i]],std_mV=float(std[i]),peak_to_peak_mV=float(ptp[i])) for i in idx]
with (OUT/'all_neuron_ranking_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
selected=np.r_[largest,smallest];np.savez_compressed(OUT/'selected_traces.npz',time_ms=t,nodes=selected,voltage_mV=v[:,selected],std_mV=std[selected],root_ids=node_ids[selected].astype(str),cell_types=np.array([cells[nc[i]] for i in selected]))
joint=v[window][:,selected];ylo=float(np.floor(float(joint.min())/5)*5-5);yhi=float(np.ceil(float(joint.max())/5)*5+5);ylims=(ylo,yhi);yticks=np.arange(np.ceil(ylo/20)*20,yhi+1,20)
# All 20 panels use the exact same x limits, y limits and ticks.
fig,axes=plt.subplots(5,4,figsize=(16,11),sharex=True,sharey=True,layout='constrained')
for group,ns,colbase,color in [('largest',largest,0,'#cf4438'),('smallest',smallest,2,'#2875b4')]:
 for k,node in enumerate(ns):
  row=k%5;col=colbase+k//5;ax=axes[row,col];ax.plot(t[window],v[window,node],color=color,lw=.85);ax.set_xlim(200,1200);ax.set_ylim(*ylims);ax.set_yticks(yticks);ax.tick_params(labelleft=True,labelsize=8);ax.grid(alpha=.17)
  sd=f'{std[node]:.3f}' if std[node]>=.001 else f'{std[node]:.2g}'
  ax.set_title(f'#{k+1} {cells[nc[node]]} | node {node}\nSD {sd} mV; range {ptp[node]:.3f} mV',fontsize=9,color=color)
  if row==4:ax.set_xlabel('Observation time (ms)',fontsize=9)
  if col in [0,2]:ax.set_ylabel('Raw V (mV)',fontsize=9)
for ax in axes.flat:
 assert ax.get_xlim()==(200.,1200.) and ax.get_ylim()==ylims
 assert np.array_equal(ax.get_yticks(),yticks)
fig.suptitle('10 largest fluctuations (left two columns)                         10 smallest fluctuations (right two columns)\n34401154 | dt=1 ms | constant background 20 pA at photoreceptors | original initialization\nRanked across all neurons by voltage SD over 200-1200 ms; identical axes on every panel; no smoothing',fontsize=12)
fig.savefig(OUT/'largest10_smallest10_same_axes.png',dpi=150);plt.close(fig)
# Separate full-size sheets for inspection, also on the same global axes.
for group,ns,color in [('largest',largest,'#cf4438'),('smallest',smallest,'#2875b4')]:
 fig,axes=plt.subplots(5,2,figsize=(11,11),sharex=True,sharey=True,layout='constrained')
 for k,node in enumerate(ns):
  ax=axes[k%5,k//5];ax.plot(t[window],v[window,node],color=color,lw=1.);ax.set_xlim(200,1200);ax.set_ylim(*ylims);ax.set_yticks(yticks);ax.tick_params(labelleft=True);ax.grid(alpha=.17);ax.set_title(f'#{k+1} {cells[nc[node]]} | node {node} | SD={std[node]:.4g} mV',fontsize=10)
  if k%5==4:ax.set_xlabel('Observation time (ms)')
  ax.set_ylabel('Raw V (mV)')
 fig.suptitle(f'10 {group} voltage fluctuations | same global x/y scale\n34401154, dt=1 ms, constant background, rank window 200-1200 ms',fontsize=12);fig.savefig(OUT/f'{group}10_same_axes.png',dpi=150);plt.close(fig)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
manifest=dict(run=str(run),initial_state='original_initial_state.npz, after the original 324-ms background stage',dt_ms=1.,stimulation='constant 20 pA at photoreceptors, zero direct current at other neurons; no added pulse',ranking='all neurons, SD in float64 after centering at the first sample; ties broken by ascending node index',rank_window_ms=[200,1200],n_neurons=N,n_exactly_constant=int((ptp==0).sum()),same_global_axes=dict(xlim=[200,1200],ylim=list(ylims),yticks=yticks.tolist()),saved_trace_reproduction_max_error_mV=reference_error,source_sha256=hashes,source_unchanged=True,selected=rows)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2),flush=True)
