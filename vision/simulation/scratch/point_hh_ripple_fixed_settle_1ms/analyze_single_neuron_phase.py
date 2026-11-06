"""Analyze saved single-neuron states and verify the 8000-ms time continuation."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
from scipy.signal import find_peaks
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent;OUT=BASE/'single_neuron_phase_analysis';OUT.mkdir(exist_ok=True)
manifest=json.loads((BASE/'manifest.json').read_text());hash_file=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert all(hash_file(p)==h for p,h in manifest['source_sha256'].items())
rows=[];state_rows=[];results={}
for run_name in ['latest_34401154','original_34400581']:
 root=BASE/run_name;q=np.load(root/'comparison_traces.npz');sa=np.load(root/'original_initial_state.npz');sb=np.load(root/'after_segmented_settle_state.npz');t=q['time_ms'];labels=q['representative_labels'].tolist();tail=(t>=200)&(t<=1200);info=[]
 for j,label in enumerate(labels):
  node=int(q['representative_nodes'][j]);cell=label.split('_')[0];a=q['representative_before'][:,j].astype(float);b=q['representative_after'][:,j].astype(float)
  row=dict(run=run_name,label=label,cell=cell,node=node,voltage_initial_A_mV=float(sa['v'][node]),voltage_initial_B_mV=float(sb['v'][node]),voltage_initial_B_minus_A_mV=float(sb['v'][node]-sa['v'][node]))
  for key in ['v','HH_m','HH_h','HH_n']:state_rows.append(dict(run=run_name,label=label,node=node,state=key,A=float(sa[key][node]),B=float(sb[key][node]),B_minus_A=float(sb[key][node]-sa[key][node])))
  peaks={}
  for name,y in [('A',a),('B',b)]:
   amp=float(np.ptp(y[tail]));pk,_=find_peaks(y,prominence=max(.05,.2*amp),distance=5);pt=t[pk];hit=(pt>=200)&(pt<=1200);peak_t=pt[hit];period=np.diff(peak_t);peaks[name]=pt.tolist()
   row.update({name+'_min_mV':float(y[tail].min()),name+'_max_mV':float(y[tail].max()),name+'_peak_to_peak_mV':amp,name+'_mean_mV':float(y[tail].mean()),name+'_first_peak_ms':int(pt[0]) if len(pt) else None,name+'_period_mean_ms':float(period.mean()) if len(period) else None,name+'_period_std_ms':float(period.std()) if len(period) else None,name+'_n_peaks':len(peak_t)})
  if row['A_period_mean_ms'] is not None and np.ptp(a[tail])>=.1 and np.ptp(b[tail])>=.1:
   maxlag=int(np.ceil(row['A_period_mean_ms']/2));candidates=[]
   for lag in range(-maxlag,maxlag+1):
    k=np.flatnonzero(tail);k=k[(k+lag>=0)&(k+lag<len(t))];candidates.append((float(np.corrcoef(a[k+lag],b[k])[0,1]),lag,k))
   corr,lag,k=max(candidates,key=lambda x:x[0]);aligned=b[k]-a[k+lag]
   row.update(alignment_definition='B(t) compared with A(t+lag)',best_lag_ms=lag,phase_lead_degrees=float(lag/row['A_period_mean_ms']*360),unshifted_correlation=float(np.corrcoef(a[tail],b[tail])[0,1]),aligned_correlation=corr,unaligned_rmse_mV=float(np.sqrt(np.mean((a[k]-b[k])**2))),aligned_rmse_mV=float(np.sqrt(np.mean(aligned**2))),aligned_max_abs_error_mV=float(abs(aligned).max()))
  else:row.update(alignment_definition='No resolved periodic waveform',best_lag_ms=None,phase_lead_degrees=None,unshifted_correlation=None,aligned_correlation=None,unaligned_rmse_mV=None,aligned_rmse_mV=None,aligned_max_abs_error_mV=None)
  row['settle_prefix_matches_A_bitwise']=bool(np.array_equal(q['settle_representative'][:len(t),j],q['representative_before'][:,j]));row['settle_endpoint_equals_B0']=bool(q['settle_representative'][-1,j]==b[0]);assert row['settle_prefix_matches_A_bitwise'] and row['settle_endpoint_equals_B0']
  rows.append(row);info.append(dict(metrics=row,peak_times_ms=peaks))
 results[run_name]=dict(representatives=info,all_neurons_initial_voltage_diff_max_mV=float(np.max(abs(sb['v'].astype(float)-sa['v']))),all_neurons_initial_voltage_diff_rms_mV=float(np.sqrt(np.mean((sb['v'].astype(float)-sa['v'])**2))))
# Save all metrics without rounding; display figures use raw V, no smoothing.
def write_csv(path,records):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
write_csv(OUT/'single_neuron_metrics.csv',rows);write_csv(OUT/'initial_state_values.csv',state_rows)
# Directly verify that the final state is ordinary uninterrupted integration for 8000 steps.
SIM=Path('/home/yitai/drosophila_vision/vision/simulation');sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron import point_hh
root=BASE/'latest_34401154';sa=np.load(root/'original_initial_state.npz');sb=np.load(root/'after_segmented_settle_state.npz');summary=json.loads((root/'summary.json').read_text());run=Path(summary['run']);opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
torch.set_num_threads(1);session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);p=train.params_from_z(z,session);I=session.packs['spread']['bright'].i_sti[:,0,:];pp=point_hh.prepare_params(p,I,session)
state_map={'v':'v','m':'HH_m','h':'HH_h','n':'HH_n','s':'Syn_s'};st={k:torch.as_tensor(sa[rk])[None,:].clone() for k,rk in state_map.items()};state_at_20=None
with torch.no_grad():
 for step in range(1,8001):
  st=point_hh.step(st,pp,I,session,delta_ms=1.)
  if step==20:state_at_20={k:v[0].numpy().copy() for k,v in st.items()}
verification={rk:dict(bitwise_equal=bool(np.array_equal(st[k][0].numpy(),sb[rk])),max_abs_error=float(np.max(abs(st[k][0].numpy().astype(float)-sb[rk])))) for k,rk in state_map.items()}
assert all(v['bitwise_equal'] for v in verification.values()),verification
node=5243;mi1_state20={rk:dict(A_at_20ms=float(state_at_20[k][node]),B_at_0ms=float(sb[rk][node]),difference=float(sb[rk][node]-state_at_20[k][node])) for k,rk in state_map.items() if k!='s'}
results['continuation_verification']=dict(dt_ms=1.,steps=8000,all_state_fields=verification,Mi1_node_5243_state_at_shift=mi1_state20)
(OUT/'analysis_summary.json').write_text(json.dumps(results,indent=2)+'\n')
# Primary example: the same Mi1 neuron, with and without the measured phase correction.
q=np.load(root/'comparison_traces.npz');j=q['representative_labels'].tolist().index('Mi1_largest_original_oscillation');a=q['representative_before'][:,j];b=q['representative_after'][:,j];t=q['time_ms'];lag=20
fig,axes=plt.subplots(1,2,figsize=(12,4.3),sharex=True,sharey=True,layout='constrained');mask=(t>=0)&(t<=126)
axes[0].plot(t[mask],a[mask],color='#2874b6',label='A: original initial state',lw=1.6);axes[0].plot(t[mask],b[mask],color='#d24b36',label='B: after +8000 ms',lw=1.6)
axes[0].scatter([0,0],[a[0],b[0]],c=['#2874b6','#d24b36'],s=35,zorder=4);axes[0].set_title('Same observation clock: initial voltages differ');axes[0].text(.02,.05,f'A(0) = {a[0]:.4f} mV\nB(0) = {b[0]:.4f} mV',transform=axes[0].transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
k=np.flatnonzero(mask);axes[1].plot(t[k],a[k+lag],color='#2874b6',lw=2.5,label='A(t + 20 ms)');axes[1].plot(t[k],b[k],color='#d24b36',lw=1.2,ls='--',label='B(t)');axes[1].set_title('After 20 ms phase alignment: almost identical')
for ax in axes:
 ax.set_xlim(0,126);ax.set_ylim(-55,-5);ax.set_xticks(np.arange(0,127,21));ax.set_yticks(np.arange(-50,0,10));ax.tick_params(labelleft=True);ax.set_xlabel('Observation time (ms)');ax.set_ylabel('Raw membrane voltage (mV)');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='upper right')
fig.suptitle('Same Mi1 neuron #5243 | dt = 1 ms | period = 42 ms\n8000 ms = 190 cycles + 20 ms; no parameter changes',fontsize=12);fig.savefig(OUT/'Mi1_exact_phase_alignment.png',dpi=160);plt.close(fig)
# Corresponding alignment plots for all three previously displayed neurons.
fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True,sharey='row',layout='constrained')
for ri,cell in enumerate(['L1','Mi1','Tm3']):
 j=q['representative_labels'].tolist().index(cell+'_largest_original_oscillation');node=int(q['representative_nodes'][j]);r=next(r for r in rows if r['run']=='latest_34401154' and r['node']==node);lag=r['best_lag_ms'];a=q['representative_before'][:,j];b=q['representative_after'][:,j];k=np.flatnonzero((t>=200)&(t<=450));k=k[(k+lag>=0)&(k+lag<len(t))];xx=t[k]-200;lo=min(float(a[k].min()),float(b[k].min()),float(a[k+lag].min()));hi=max(float(a[k].max()),float(b[k].max()),float(a[k+lag].max()));pad=.06*(hi-lo)
 axes[ri,0].plot(xx,a[k],color='#2874b6',lw=1.3,label='A(t)');axes[ri,0].plot(xx,b[k],color='#d24b36',lw=1.3,label='B(t)');axes[ri,1].plot(xx,a[k+lag],color='#2874b6',lw=2,label=f'A(t {lag:+d} ms)');axes[ri,1].plot(xx,b[k],color='#d24b36',lw=1.1,ls='--',label='B(t)')
 for ci in range(2):
  ax=axes[ri,ci];ax.set_ylim(lo-pad,hi+pad);ax.set_xlim(0,250);ax.set_ylabel(cell+f' #{node}\nV (mV)');ax.tick_params(labelleft=True);ax.grid(alpha=.15);ax.legend(fontsize=7)
  if ri==2:ax.set_xlabel('Time within comparison window (ms)')
axes[0,0].set_title('Unaligned');axes[0,1].set_title('Phase aligned (same x/y scales)');fig.suptitle('Single-neuron phase comparison; no waveform smoothing or amplitude rescaling',fontsize=12);fig.savefig(OUT/'three_neuron_phase_alignment.png',dpi=150);plt.close(fig)
assert all(hash_file(p)==h for p,h in manifest['source_sha256'].items())
print('FULL_STATE_CONTINUATION',verification,flush=True)
for r in rows:
 if r['run']=='latest_34401154' and r['label'].endswith('largest_original_oscillation') and r['cell'] in ['L1','Mi1','Tm3']:print(json.dumps(r),flush=True)
print('ALL_DONE',OUT,flush=True)
