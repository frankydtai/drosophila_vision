"""Temporary k_minus=0.01 override; preserve all other parameters and re-rank all neurons."""
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
from neuron.forward import forward_v
old_summary=json.loads((PARENT/'summary.json').read_text());run=Path(old_summary['run']);opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
torch.set_num_threads(1)
sources=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/4_forward.py',run/'data/best_param.npz',run/'data/train_opts.json',PARENT/'original_initial_state.npz',Path(__file__)]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);params=train.params_from_z(z,session)
unchanged={k:v.detach().clone() for k,v in params.items() if torch.is_tensor(v) and k!='k_minus_syn'}
old_k=params['k_minus_syn'].detach().clone();params['k_minus_syn']=torch.full_like(params['k_minus_syn'],.01)
assert bool((params['k_minus_syn']==.01).all())
pack=session.packs['spread']['bright'];I=pack.i_sti[:,0,:];pp=point_hh.prepare_params(params,I,session);N=session.connectome.n_node
warmup=round(float(opts['spread_sti_opts']['ms_pre']))
with torch.no_grad():
 full=forward_v(session,params,I[:,None,:].expand(1,warmup+1201,N),pack=pack)
assert torch.isfinite(full).all()
v=full[0,warmup:].numpy();t=np.arange(1201);window=(t>=200)&(t<=1200)
assert v.shape==(1201,N)
assert all(torch.equal(params[k],value) for k,value in unchanged.items())
assert bool((pp['k_minus_syn']==.01).all())

# Arithmetic mean of every neuron, equal weight per neuron, no smoothing.
previous=np.load(OUT/'selected_traces.npz',allow_pickle=False)
assert N==5789
assert np.array_equal(v[:,previous['nodes']],previous['voltage_mV'])
mean=v.mean(axis=1,dtype=np.float64)
assert mean.shape==(1201,) and np.isfinite(mean).all()
assert np.allclose(mean,np.sum(v.astype(np.float64),axis=1)/N,rtol=0,atol=1e-12)
np.savez_compressed(OUT/'all_neuron_mean_traces.npz',time_ms=t,mean_voltage_mV=mean,n_neurons=N,k_minus_syn_per_ms=.01,dt_ms=1.)
np.savetxt(OUT/'all_neuron_mean_traces.csv',np.column_stack([t,mean]),delimiter=',',header='time_ms,all_5789_neurons_mean_voltage_mV',comments='')
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
ax.plot(t[window],mean[window],color='#225b9a',lw=1.25)
ax.set(xlim=(200,1200),xlabel='Observation time (ms)',ylabel='Mean membrane voltage (mV)',title='Arithmetic mean of all 5789 neurons\nk_minus_syn = 0.01 /ms | dt = 1 ms | constant background | no smoothing')
ax.grid(alpha=.22)
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
fig.savefig(OUT/'all_5789_neurons_mean_kminus001_1ms.png',dpi=170)
plt.close(fig)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
manifest=dict(n_neurons=N,mean_definition='Equal-weight arithmetic mean across all 5789 neurons at each time sample; no smoothing',k_minus_syn_per_ms=.01,dt_ms=1.,display_window_ms=[200,1200],background='20 pA at photoreceptors; no extra stimulus',selected_previous_traces_exact_match=True,source_unchanged=True,source_sha256=hashes,mean_min_mV=float(mean[window].min()),mean_max_mV=float(mean[window].max()))
(OUT/'all_neuron_mean_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2),flush=True)
