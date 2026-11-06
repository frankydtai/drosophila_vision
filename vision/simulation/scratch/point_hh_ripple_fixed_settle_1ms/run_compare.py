"""Temporary Ripple fixed-threshold settling adapter for the unchanged Point HH kernel."""
from pathlib import Path
from types import SimpleNamespace
import sys,importlib.util,json,csv,hashlib,time
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
from neuron import point_hh
from neuron.forward import forward_v
RIPPLE=Path('/home/yitai/Ripple/experiments/paper3_gene_mapping/fixed_threshold_settle.py')
spec=importlib.util.spec_from_file_location('ripple_fixed_threshold_method_readonly',RIPPLE);ripple=importlib.util.module_from_spec(spec);spec.loader.exec_module(ripple)
torch.set_num_threads(1)
RUNS={'latest_34401154':next((SIM/'0_runs/point_hh').glob('34401154-*')),'original_34400581':next((SIM/'0_runs/point_hh').glob('34400581-*'))}
RULE=dict(dt_ms=1.,block_ms=100.,max_blocks=80,min_blocks=2,voltage_tol_mV=.02,state_tol=.002,release_tol=.002,voltage_bounds_mV=[-120.,50.],observation_ms=1200,threshold_policy='fixed_absolute',occupancy_control=False)
SOURCES=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/4_forward.py',SIM/'4_train/1_param.py',SIM/'4_train/2_session.py',SIM/'1_neuron/3_schema.py',RIPPLE,Path(__file__)]
for run in RUNS.values():SOURCES.extend([run/'data/train_opts.json',run/'data/best_param.npz'])
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before_hash={str(p):hash_file(p) for p in SOURCES}
all_results={}
state_map={'v':'v','m':'HH_m','h':'HH_h','n':'HH_n','s':'Syn_s'}
def to_ripple(st):return {rn:st[pn][0].detach().cpu().numpy().copy() for pn,rn in state_map.items()}
def to_point(st):return {pn:torch.as_tensor(st[rn]).unsqueeze(0) for pn,rn in state_map.items()}
def array_hashes(params):
 return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in params.items() if torch.is_tensor(v)}
def save_csv(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

with torch.no_grad():
 for label,run in RUNS.items():
  sub=OUT/label;sub.mkdir(exist_ok=True);print('START',label,flush=True);start=time.perf_counter()
  opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
  for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
  session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);params=train.params_from_z(z,session);pack=session.packs['spread']['bright']
  assert session.delta_ms==1. and session.delta_ms_pre==1.
  I=pack.i_sti[:,0,:].clone();pp=point_hh.prepare_params(params,I,session);phash=array_hashes(pp);cells=list(session.connectome.cells);nc=session.connectome.node_cells;nc_np=nc.numpy();N=session.connectome.n_node;counts=torch.bincount(nc,minlength=len(cells));src=session.connectome.conn.source_idxs.numpy();tar=session.connectome.conn.target_idxs.numpy()
  def integrate(initial,n_steps):
   state={k:v.clone() for k,v in initial.items()};voltage=torch.empty((n_steps+1,N),dtype=I.dtype);voltage[0]=state['v'][0];slo=float(state['s'].min());shi=float(state['s'].max())
   for ti in range(1,n_steps+1):
    state=point_hh.step(state,pp,I,session,delta_ms=1.);voltage[ti]=state['v'][0];slo=min(slo,float(state['s'].min()));shi=max(shi,float(state['s'].max()))
   assert all(torch.isfinite(value).all() for value in state.values())
   assert all(bool(((state[k]>=0)&(state[k]<=1)).all()) for k in ['m','h','n','s'])
   return state,voltage,np.array([float(voltage.min()),float(voltage.max()),float((voltage[-1]-voltage[-min(21,len(voltage))]).abs().max()),slo,shi])
  def means(voltage):
   out=torch.zeros((len(voltage),len(cells)),dtype=voltage.dtype);out.scatter_add_(1,nc.unsqueeze(0).expand(len(voltage),-1),voltage);return (out/counts).numpy()
  # Original initialization, including its usual pre-stimulus background period.
  seed=point_hh.pre_steady(session,pp,1,i_sti=I[:,None,:]);warm_ms=float(opts['spread_sti_opts']['ms_pre']);original,_,_=integrate(seed,round(warm_ms))
  # Check the adapter's original initial state against the actual production driver.
  ref=forward_v(session,params,I[:,None,:].expand(1,round(warm_ms)+1,N),pack=pack)
  original_driver_error=float((ref[:,-1]-original['v']).abs().max());assert original_driver_error<2e-5,original_driver_error
  np.savez_compressed(sub/'original_initial_state.npz',**to_ripple(original))
  _,vb,safety_before=integrate(original,RULE['observation_ms']);mb=means(vb);vbn=vb.numpy()
  # Save population means plus central and most fluctuating single neurons.
  tt=np.arange(len(vb));tail=(tt>=200)&(tt<=1200);std_before=np.std(vbn[tail],axis=0);pp_before=np.ptp(vbn[tail],axis=0)
  reps=[];rep_labels=[]
  for cell in ['L1','Mi1','Tm3','Mi4','Mi9','T4a']:
   ns=np.flatnonzero(nc_np==cells.index(cell));us=np.asarray(session.connectome.us)[ns];vs=np.asarray(session.connectome.vs)[ns];center=int(ns[np.argmin(us*us+vs*vs)]);worst=int(ns[np.argmax(std_before[ns])]);reps.extend([center,worst]);rep_labels.extend([cell+'_central',cell+'_largest_original_oscillation'])
  reps=np.array(reps,dtype=int);reps_saved_before=vbn[:,reps].copy()
  settle_means=[];settle_reps=[];block_count=[0]
  class FixedMapper:
   threshold_policy='fixed_absolute';synapse_name='Syn';pre_index=src;delta=pp['delta_syn'].numpy()
   def parameters(self,beta,threshold,anchor_beta,xp=np):return None,None,pp['v_th_syn'].numpy().copy(),None
  mapper=FixedMapper();frozen_threshold=pp['v_th_syn'].numpy().copy()
  def block_integrator(beta,states,threshold,anchor_beta,current):
   assert np.array_equal(threshold,frozen_threshold);assert np.array_equal(np.asarray(current),I[0].numpy())
   after,voltage,safety=integrate(to_point(states),100);mm=means(voltage);rv=voltage[:,reps].numpy().copy();settle_means.append(mm if not settle_means else mm[1:]);settle_reps.append(rv if not settle_reps else rv[1:]);block_count[0]+=1
   if block_count[0]<=2 or block_count[0]%10==0:print('SETTLE',label,'block',block_count[0],'tail_drift_mV',safety[2],flush=True)
   return mm.T,safety,to_ripple(after)
  after_ripple,diagnostic=ripple.fixed_threshold_settle(beta=np.empty(0),states=to_ripple(original),threshold=frozen_threshold,mapper=mapper,settle=block_integrator,current=I[0].numpy(),max_blocks=80,min_blocks=2,voltage_tol=.02,state_tol=.002,release_tol=.002,voltage_min=-120.,voltage_max=50.)
  assert array_hashes(pp)==phash,'Physical parameter changed during settling'
  assert np.array_equal(pp['v_th_syn'].numpy(),frozen_threshold)
  for row in diagnostic['history']:
   node=row['voltage_drift_cell_index'];edge=row['release_residual_edge_index'];row.update(worst_voltage_cell_type=cells[nc_np[node]],worst_voltage_node=node,worst_release_pre_type=cells[nc_np[src[edge]]],worst_release_post_type=cells[nc_np[tar[edge]]],worst_release_gS_nS=float(pp['gS'][edge]))
  save_csv(sub/'settle_history.csv',diagnostic['history']);(sub/'settle_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n');np.savez_compressed(sub/'after_segmented_settle_state.npz',**after_ripple)
  _,va,safety_after=integrate(to_point(after_ripple),RULE['observation_ms']);ma=means(va);van=va.numpy();std_after=np.std(van[tail],axis=0);pp_after=np.ptp(van[tail],axis=0)
  all_node_rows=[dict(node=i,cell_type=cells[nc_np[i]],before_std_mV=float(std_before[i]),after_std_mV=float(std_after[i]),before_peak_to_peak_mV=float(pp_before[i]),after_peak_to_peak_mV=float(pp_after[i])) for i in range(N)];save_csv(sub/'all_neuron_oscillation_metrics.csv',all_node_rows)
  metrics=[]
  for cell in cells:
   ci=cells.index(cell);hit=nc_np==ci
   metrics.append(dict(cell=cell,n_neurons=int(hit.sum()),before_population_std_mV=float(np.std(mb[tail,ci])),after_population_std_mV=float(np.std(ma[tail,ci])),before_population_pp_mV=float(np.ptp(mb[tail,ci])),after_population_pp_mV=float(np.ptp(ma[tail,ci])),before_node_std_median_mV=float(np.median(std_before[hit])),after_node_std_median_mV=float(np.median(std_after[hit])),before_node_pp_p95_mV=float(np.percentile(pp_before[hit],95)),after_node_pp_p95_mV=float(np.percentile(pp_after[hit],95))))
  save_csv(sub/'population_oscillation_metrics.csv',metrics)
  np.savez_compressed(sub/'comparison_traces.npz',time_ms=tt,cells=np.array(cells),population_before=mb,population_after=ma,representative_nodes=reps,representative_labels=np.array(rep_labels),representative_before=reps_saved_before,representative_after=van[:,reps],settle_time_ms=np.arange(sum(len(v) for v in settle_means)),settle_population=np.concatenate(settle_means),settle_representative=np.concatenate(settle_reps))
  info=dict(run=str(run),rules=RULE,fp=opts.get('fp'),original_presteady=dict(mode=opts['pre_steady'],iterations=opts['pre_steady_n_iter'],damping=opts['pre_steady_damp'],background_warmup_ms=warm_ms),k_minus_range=[float(pp['k_minus_syn'].min()),float(pp['k_minus_syn'].max())],v_th_range=[float(pp['v_th_syn'].min()),float(pp['v_th_syn'].max())],status=diagnostic['status'],converged=diagnostic['converged'],blocks=diagnostic['blocks'],additional_settle_ms=diagnostic['blocks']*100,original_driver_max_error_mV=original_driver_error,parameters_unchanged=True,threshold_unchanged=True,last_block=diagnostic['history'][-1],metrics=metrics,before_safety=safety_before.tolist(),after_safety=safety_after.tolist(),elapsed_s=time.perf_counter()-start)
  all_results[label]=info;(sub/'summary.json').write_text(json.dumps(info,indent=2)+'\n');(OUT/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n');print('DONE',label,info['status'],info['last_block'],flush=True)
# Keep source, parameter and saved train_opts identities auditable.
after_hash={str(p):hash_file(p) for p in SOURCES};assert before_hash==after_hash,'An input source or saved checkpoint changed during this experiment'
(OUT/'manifest.json').write_text(json.dumps(dict(rule=RULE,source_sha256=before_hash,input_sources_unchanged=True,output_role='temporary diagnostic; never promote budget expiry to convergence'),indent=2)+'\n')
# Compare raw constant-background trajectories; do not subtract a simulated control.
for label,info in all_results.items():
 sub=OUT/label;q=np.load(sub/'comparison_traces.npz');cells=q['cells'].tolist();fig,axes=plt.subplots(3,2,figsize=(12,8.5),sharex=True)
 for row,cell in enumerate(['L1','Mi1','Tm3']):
  ci=cells.index(cell);nodepos=q['representative_labels'].tolist().index(cell+'_largest_original_oscillation');ni=int(q['representative_nodes'][nodepos])
  for col in range(2):
   ax=axes[row,col]
   if col==0:yb=q['population_before'][:,ci];ya=q['population_after'][:,ci];ax.set_title(cell+' population mean')
   else:yb=q['representative_before'][:,nodepos];ya=q['representative_after'][:,nodepos];ax.set_title(cell+f' same neuron #{ni} (selected before settling)')
   mask=(q['time_ms']>=200)&(q['time_ms']<=800);ax.plot(q['time_ms'][mask],yb[mask],color='#2874b6',lw=1.1,label='Original initialization');ax.plot(q['time_ms'][mask],ya[mask],color='#d24b36',lw=1.1,label=f'After {info["additional_settle_ms"]:g} ms segmented settling');ax.set_ylabel('Raw membrane V (mV)');ax.grid(alpha=.15)
   if row==2:ax.set_xlabel('Time from start of observation (ms)')
 axes[0,0].legend(fontsize=7,frameon=False);fig.suptitle(f'{label}: Ripple fixed-threshold segmented settling | dt = 1 ms\nConstant background, unchanged parameters/thresholds | {info["status"]} after {info["blocks"]} blocks',fontsize=12);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(sub/'oscillation_before_after.png',dpi=150);plt.close(fig)
# Progress against the exact acceptance limits.
fig,axes=plt.subplots(2,2,figsize=(11,6.5),sharex=True)
for label in all_results:
 d=json.loads((OUT/label/'settle_diagnostic.json').read_text());hist=d['history'];x=[r['block']*100 for r in hist]
 for ax,key,tol,title in zip(axes.flat,['voltage_drift_mv','tail_voltage_drift_mv','gate_drift','release_residual'],[.02,.02,.002,.002],['Block endpoint voltage drift','Last-20-step voltage drift','Gate / release state drift','Release equilibrium residual |s-s_inf|']):
  ax.semilogy(x,[max(r[key],1e-12) for r in hist],label=label);ax.axhline(tol,color='black',ls='--',lw=.8);ax.set_title(title);ax.grid(alpha=.15)
for ax in axes[-1]:ax.set_xlabel('Additional background integration (ms)')
axes[0,0].legend(fontsize=7);fig.suptitle('Ripple convergence checks: dashed line = required upper bound',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'convergence_checks.png',dpi=150)
print('ALL_DONE',OUT,flush=True)
