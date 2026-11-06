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
from neuron.forward import forward_v
run=next((SIM/'0_runs/hp_lp').glob('34381872-*'));opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
opts['spread_sti_opts']['ms_response']=1200.;torch.set_num_threads(1)
sources=[SIM/'1_neuron/1_2_hp_lp.py',SIM/'1_neuron/4_forward.py',run/'data/best_param.npz',run/'data/train_opts.json',Path(__file__)]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);params=train.params_from_z(z,session);pack=session.packs['spread']['bright'];N=session.connectome.n_node
assert session.delta_ms==1. and session.delta_ms_pre==1.
I=pack.i_sti[:,0,:].clone()
with torch.no_grad():
 full=forward_v(session,params,I[:,None,:].expand_as(pack.i_sti),pack=pack)
assert torch.isfinite(full).all()
voltage=full[0,int(pack.t_onset):int(pack.t_onset)+1201];v=voltage.numpy();t=np.arange(len(v));assert len(t)==1201
window=(t>=200)&(t<=1200)
# Match the earlier saved HPLP background data, using the same population reduction.
old=np.load(SIM/'scratch/point_hh_k_minus_1ms_34400581/hp_lp.npz');reference_error=0.
nc_reference=session.connectome.node_cells
for cell in ['L1','Mi1','Tm3']:
 hit=nc_reference==list(session.connectome.cells).index(cell)
 mean=voltage[:,hit].mean(-1).numpy();reference=old[cell+'_control'][old['time_ms']>=0]
 reference_error=max(reference_error,float(np.max(abs(mean-reference[:1201]))))
assert reference_error<2e-5,reference_error

previous=np.load(OUT/'selected_traces.npz',allow_pickle=False)
assert N==5789 and v.shape==(1201,N)
assert np.array_equal(v[:,previous['nodes']],previous['voltage_mV'])
mean=v.mean(axis=1,dtype=np.float64)
assert mean.shape==(1201,) and np.isfinite(mean).all()
assert np.allclose(mean,np.sum(v.astype(np.float64),axis=1)/N,rtol=0,atol=1e-12)
np.savez_compressed(OUT/'all_neuron_mean_traces.npz',time_ms=t,mean_voltage_mV=mean,n_neurons=N,dt_ms=1.)
np.savetxt(OUT/'all_neuron_mean_traces.csv',np.column_stack([t,mean]),delimiter=',',header='time_ms,all_5789_neurons_mean_voltage_mV',comments='')
fig,ax=plt.subplots(figsize=(12,4.8),layout='constrained')
ax.plot(t[window],mean[window],color='#277f47',lw=1.3)
lo,hi=float(mean[window].min()),float(mean[window].max())
center=(lo+hi)/2;span=max(.3,(hi-lo)*1.15)
ax.set(xlim=(200,1200),ylim=(center-span/2,center+span/2),xlabel='Observation time (ms)',ylabel='Mean membrane voltage (mV)',title='HP-LP: arithmetic mean of all 5789 neurons\ndt = 1 ms | constant background: 20 pA at photoreceptors | no smoothing')
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.grid(alpha=.22)
fig.savefig(OUT/'all_5789_neurons_mean_hplp_1ms.png',dpi=170);plt.close(fig)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
manifest=dict(run=str(run),n_neurons=N,mean_definition='Equal-weight arithmetic mean across all neurons at each sample, no smoothing',dt_ms=1.,display_window_ms=[200,1200],background='20 pA at photoreceptors, zero direct current elsewhere, no added pulse',warmup_ms=float(opts['spread_sti_opts']['ms_pre']),previous_selected_traces_exact_match=True,previous_population_reference_max_error_mV=reference_error,mean_min_mV=lo,mean_max_mV=hi,peak_to_peak_mV=hi-lo,source_unchanged=True,source_sha256=hashes)
(OUT/'all_neuron_mean_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2),flush=True)
