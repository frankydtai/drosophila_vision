from pathlib import Path
import json,time
import numpy as np
import torch
from scipy.signal import find_peaks
from stability import load,OUT
from neuron import point_hh
session,p,I,st=load();q=np.load(OUT/'dc_equilibrium.npz');mode=np.load(OUT/'unstable_mode_k_0.1_sigma_0.3.npz');N=len(q['v']);lam=complex(mode['eigenvalue']);u=mode['eigenvector'];active=mode['active'];source=session.connectome.conn.source_idxs;nc=session.connectome.node_cells.numpy();cells=list(session.connectome.cells)
mi=np.flatnonzero(nc==cells.index('Mi1'));node=int(mi[np.argmax(abs(u[mi]))]);u=u*np.exp(-1j*np.angle(u[node]));scale=abs(u[:N]).max();eps=1e-4;u=u/scale
basev=torch.from_numpy(q['v']);m=torch.from_numpy(q['m']);h=torch.from_numpy(q['h']);n=torch.from_numpy(q['n']);s=point_hh.release_steady(basev[source],p['v_th_syn'],p['delta_syn'])
p['k_minus_syn']=torch.full_like(p['k_minus_syn'],.1);pp=point_hh.prepare_params(p,basev[None,:].expand(3,-1),session)
statebase={'v':basev,'m':m,'h':h,'n':n,'s':s};init={k:value[None,:].repeat(3,1) for k,value in statebase.items()}
for j,k in enumerate(['v','m','h','n']):
 perturb=torch.from_numpy(eps*u[j*N:(j+1)*N].real)
 init[k][0]+=perturb;init[k][2]+=perturb
sd=torch.zeros_like(s);sd[torch.as_tensor(active)]=torch.from_numpy(eps*u[4*N:].real);init['s'][0]+=sd
original=point_hh._release_step
def freeze_third(*args,**kwargs):
 value=original(*args,**kwargs);return torch.cat((value[:2],s[None,:]),0)
point_hh._release_step=freeze_third
results={}
try:
 with torch.no_grad():
  for dt in [1.,.025]:
   print('START dt',dt,flush=True);state={k:v.clone() for k,v in init.items()};tt=np.arange(round(300/dt)+1)*dt;record=[];times=[];stride=max(1,round(.1/dt));drive=I[None,:].expand(3,-1)
   for i,t in enumerate(tt):
    if i%stride==0:
     record.append([float(state['v'][0,node]-state['v'][1,node]),float(state['v'][2,node]-basev[node]),float((state['v'][1]-basev).abs().max())]);times.append(t)
    if i==len(tt)-1:break
    state=point_hh.step(state,pp,drive,session,delta_ms=dt)
   data=np.asarray(record);times=np.asarray(times);analytic=eps*np.real(u[node]*np.exp(lam*times));results[str(dt)]={'time_ms':times,'perturbed_v_mV':data[:,0],'fixed_synapse_v_mV':data[:,1],'dc_drift_max_mV':data[:,2],'linear_prediction_mV':analytic}
   assert np.isfinite(data).all();np.savez_compressed(OUT/f'constant_light_perturb_dt_{dt:g}.npz',**results[str(dt)])
   print('DONE',dt,'max response',abs(data[:,0]).max(),flush=True)
finally:point_hh._release_step=original
r=results['0.025'];t=r['time_ms'];v=r['perturbed_v_mV'];peaks,_=find_peaks(abs(v));use=peaks[(t[peaks]>=25)&(t[peaks]<=210)];fit=np.polyfit(t[use],np.log(abs(v[use])),1)
within=t<=180;error=float(np.linalg.norm(v[within]-r['linear_prediction_mV'][within])/np.linalg.norm(r['linear_prediction_mV'][within]));frozen60=float(abs(r['fixed_synapse_v_mV'][t>=60]).max())
summary=dict(background='constant throughout, no moving bar and no light switches',chosen_cell='Mi1',chosen_node=node,initial_max_voltage_perturb_mV=eps,eigenvalue_real_per_ms=lam.real,eigenvalue_imag_per_ms=lam.imag,predicted_growth_per_ms=lam.real,measured_growth_per_ms=float(fit[0]),linear_prediction_relative_error_first_180ms=error,fixed_synapse_max_response_after_60ms_mV=frozen60,unperturbed_equilibrium_max_drift_mV=float(r['dc_drift_max_mV'].max()),peak_times_ms=t[use].tolist(),peak_amplitudes_mV=abs(v[use]).tolist())
(OUT/'demonstration_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
