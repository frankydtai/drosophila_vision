"""Independent standard-forward validation and reloadable parameter bundles."""
from pathlib import Path
from types import SimpleNamespace
import json,csv,gc,hashlib
import numpy as np
import torch
import search as s
from neuron.forward import forward_v
from train.implementation import load_best_node_vals,session_z_from_node_vals
from train.cost import forward_pack,pack_traces,gt_affine_from_pack,_pack_cost_parts_from_v_readout,_weighted_mean_cost_parts
from figure.mbar import _traces_from_forward
from task.mbar.pack import bar_specs_from_task,dsi_values
import task.mbar.pack as mp
OUT=Path(__file__).resolve().parent;VAL=OUT/'validation';VAL.mkdir(exist_ok=True)
old_signals=mp.build_mbar_signals
def uncached(*args,**kwargs):kwargs['use_cache']=False;return old_signals(*args,**kwargs)
mp.build_mbar_signals=uncached
CASES={'reference':dict(name='reference',forward_gain=1.),'gain3':dict(name='gain3',forward_gain=3.,forward_threshold_mV=-51.,forward_delta_mV=.5),'gain6':dict(name='gain6',forward_gain=6.,forward_threshold_mV=-51.,forward_delta_mV=.5)}

def params(label):
 p={k:(v.clone() if torch.is_tensor(v) else v) for k,v in s.BASE.items()};case=CASES[label]
 p['gS'][s.FORWARD]*=case['forward_gain']
 if label!='reference':p['v_th_syn'][s.FORWARD]=-51.;p['delta_syn'][s.FORWARD]=.5
 assert torch.all(p['k_minus_syn']==.01)
 return p

checks=[];protocol_metrics=[];candidate_data={};ordinary_step=s.hh.step

def checked_step(*args,**kwargs):
 result=ordinary_step(*args,**kwargs)
 if checked_step.counter%100==0:
  assert all(torch.isfinite(value).all() for value in result.values())
  assert all(((result[k]>=0)&(result[k]<=1)).all() for k in ['m','h','n','s'])
 checked_step.counter+=1;return result
checked_step.counter=0

@torch.no_grad()
def protocol(label,name,warmup=1000,duration=160,amp=20.,post=1500):
 total=warmup+duration+post+1;drive=s.I0[:,None,:].expand(2,total,-1).clone();drive[0,warmup:warmup+duration]=s.I0+(amp/20.)*(s.ILIGHT-s.I0)
 p=params(label);s.hh.step=checked_step
 try:
  branches=[forward_v(s.SESSION,p,drive[b:b+1],pack=SimpleNamespace(t_onset=warmup)) for b in range(2)]
  v=torch.cat(branches,dim=0)
  del branches
 finally:s.hh.step=ordinary_step
 assert torch.isfinite(v).all();assert torch.equal(v[0,:warmup],v[1,:warmup]);t=np.arange(total)-warmup
 on=(t>=0)&(t<duration);off=(t>=duration)&(t<=duration+900);late=t>=duration+post-150
 pulse=np.stack([v[0,:,s.NC==ci].mean(-1).numpy() for ci in range(len(s.CELLS))],axis=1)
 control=np.stack([v[1,:,s.NC==ci].mean(-1).numpy() for ci in range(len(s.CELLS))],axis=1);evoked=pulse-control
 rows=[];node_rows=[]
 for ci,cell in enumerate(s.CELLS):
  y=evoked[:,ci];sign=(-1 if cell in ['L1','L2','L3','Mi9'] else 1)*(1 if amp>=0 else -1);op=np.maximum(-sign*y[off],0);ot=t[off];pk=int(op.argmax());height=float(op[pk]);half=op>=height/2;lo=hi=pk
  while lo>0 and half[lo-1]:lo-=1
  while hi+1<len(half) and half[hi+1]:hi+=1
  row=dict(model=label,protocol=name,cell=cell,duration_ms=duration,current_delta_pA=amp,on_max_mV=float(y[on].max()),on_min_mV=float(y[on].min()),off_max_mV=float(y[off].max()),off_min_mV=float(y[off].min()),opposite_peak_mV=height,opposite_delay_ms=float(ot[pk]-duration),opposite_fwhm_ms=hi-lo+1 if height>0 else 0,late_evoked_max_mV=float(abs(y[late]).max()),control_late_std_mV=float(control[late,ci].std()))
  rows.append(row)
  if name=='standard' and cell in ['L1','Mi1','Tm3','T4a']:
   nodes=torch.nonzero(s.NC==ci).flatten();yy=(v[0,:,nodes]-v[1,:,nodes]).numpy();a=sign*yy[on].max(0) if sign==1 else -yy[on].min(0);b=np.maximum(-sign*yy[off],0).max(0)
   quiet_std=v[1,torch.tensor(t>=duration),:][:,nodes].std(0).numpy()
   node_rows.append(dict(cell=cell,n_nodes=len(nodes),fraction_both_peaks_at_least_3mV=float(((a>=3)&(b>=3)).mean()),opposite_peak_median_mV=float(np.median(b)),opposite_peak_p95_mV=float(np.percentile(b,95)),control_node_std_median_mV=float(np.median(quiet_std)),control_node_std_p95_mV=float(np.percentile(quiet_std,95))))
 data=dict(time_ms=t,cells=np.array(s.CELLS),pulse=pulse,control=control,evoked=evoked);np.savez_compressed(VAL/f'{label}_{name}.npz',**data)
 protocol_metrics.extend(rows);checks.append(dict(model=label,protocol=name,finite=True,pre_stimulus_matched_exact=True,voltage_min_mV=float(v.min()),voltage_max_mV=float(v.max()),node_distribution=node_rows))
 if name=='standard':candidate_data[label]=data
 print('PROTOCOL_DONE',label,name,[(r['cell'],round(r['on_max_mV'],3),round(r['off_min_mV'],3),r['opposite_fwhm_ms']) for r in rows if r['cell'] in ['Mi1','Tm3','T4a']],flush=True)
 del v,drive;gc.collect();return data

for label in ['reference','gain3','gain6']:protocol(label,'standard')
repeat=protocol('gain6','repeat');assert np.array_equal(repeat['evoked'],candidate_data['gain6']['evoked']);checks.append(dict(test='native_forward_repeat',exact=True))
for label in ['gain3','gain6']:
 for name,kw in [('long_warmup',dict(warmup=3000)),('short_pulse',dict(duration=80)),('long_pulse',dict(duration=320)),('weak_bright',dict(amp=10.)),('dark',dict(amp=-20.)),('zero',dict(amp=0.))]:protocol(label,name,**kw)

# Export the tuned existing parameters using the original session/NPZ loaders.
# No new kernel, no post-forward output transforms, and no copied Adam history.
source_npz=np.load(s.RUN/'data/best_param.npz',allow_pickle=True)
source_opts=json.loads((s.RUN/'data/train_opts.json').read_text());source_rules=Path(source_opts['point_hh']['synapse_rules'])
reload_errors={};mbar_results=[]
for label in ['reference','gain3','gain6']:
 if label=='reference':
  opts=json.loads(json.dumps(source_opts));opts['device']='cpu';sess=s.train.resolve_session(opts);sess,z=session_z_from_node_vals(*load_best_node_vals(str(s.RUN)),sess);pp=s.train.params_from_z(z,sess);pp['k_minus_syn']=torch.full_like(pp['k_minus_syn'],.01);run=None
 else:
  run=OUT/f'candidate_{label}';data=run/'data';data.mkdir(parents=True,exist_ok=True)
  lines=source_rules.read_text().splitlines(keepends=True);updated=[];changed=[]
  for line in lines:
   fields=line.split(',')
   if len(fields)>4 and (fields[0].strip(),fields[1].strip()) in [('L1','Mi1'),('L1','Tm3')]:
    fields[3]=format(float(fields[3])*CASES[label]['forward_gain'],'.17g');line=','.join(fields);changed.append((fields[0],fields[1]))
   updated.append(line)
  assert sorted(changed)==[('L1','Mi1'),('L1','Tm3')]
  rules=run/'synapse_rules.csv';rules.write_text(''.join(updated))
  opts=json.loads(json.dumps(source_opts));opts['point_hh']['synapse_rules']=str(rules);opts['params']['k_minus_syn']['mode']='fixed';opts['params']['k_minus_syn']['init']=.01
  opts['params']['v_th_syn'].setdefault('modes',{})['fixed']=['L1:Mi1','L1:Tm3']
  npz={k:source_npz[k].copy() for k in source_npz.files};pairs=list(npz['pairs']);npz['k_minus_syn'][:]=.01
  for pair in ['L1:Mi1','L1:Tm3']:
   ix=pairs.index(pair);npz['v_th_syn'][ix]=-51.;npz['delta_syn'][ix]=.5
  np.savez_compressed(data/'best_param.npz',**npz);(data/'train_opts.json').write_text(json.dumps(opts,indent=2)+'\n')
  opts['device']='cpu';sess=s.train.resolve_session(opts);sess,z=session_z_from_node_vals(*load_best_node_vals(str(run)),sess);pp=s.train.params_from_z(z,sess)
  expected=params(label);errors={}
  for key in s.COLUMNS:
   errors[key]=float((pp[key]-expected[key]).abs().max());assert torch.allclose(pp[key],expected[key],rtol=3e-6,atol=2e-5),(key,errors[key])
  reload_errors[label]=errors
  np.save(data/'train_point_hh.npy',z.numpy()[None,:]);(run/'parameter_changes.json').write_text(json.dumps(dict(source=str(s.RUN),dt_ms=1.,k_minus_syn=.01,changes=CASES[label],kernel_modified=False,optimizer_history_copied=False,physically_reloaded_max_errors=errors),indent=2)+'\n')
 with torch.no_grad():
  pack=sess.packs['mbar']['bright'];vv=forward_pack(sess,pp,pack.i_sti,pack);assert torch.isfinite(vv).all()
  parts=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,sess,*gt_affine_from_pack(pp,pack,sess),pack_traces(vv,pack),trace=vv),sess);ds={k:float(a) for k,a in dsi_values(pack,vv).items()}
 specs=bar_specs_from_task(sess,'mbar','bright');tokens=[x.token for x in specs];tr,*_=_traces_from_forward(sess,'mbar','bright',vv.numpy(),specs,tokens);curves={};metrics=[]
 for (cell,token),y in tr.ca_mean_cell.items():
  tt=np.arange(len(y))-tr.before_t[token];baseline=float(y[(tt>=-350)&(tt<=-180)].mean());yy=y-baseline;mask=(tt>=-150)&(tt<=576)
  curves[cell+'__'+token]=yy;curves['time__'+token]=tt
  metrics.append(dict(cell=cell,condition=token,positive_peak_mV=float(yy[mask].max()),negative_peak_mV=float(yy[mask].min())))
 np.savez_compressed(VAL/f'{label}_mbar.npz',**curves);cost=sum(float(a) for a in parts.values());mbar_results.append(dict(model=label,cost=cost,dsi_raw=ds,waveform_metrics=metrics))
 if run is not None:np.save(run/'data/costs.npy',np.array([cost]));(run/'evaluation.json').write_text(json.dumps(mbar_results[-1],indent=2)+'\n')
 print('MBAR_DONE',label,cost,ds,flush=True);del vv,sess,pp;gc.collect()

with (VAL/'protocol_metrics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(protocol_metrics[0]));w.writeheader();w.writerows(protocol_metrics)
summary=dict(dt_ms=1.,k_minus_syn=.01,formal_kernel_modified=False,source_run=str(s.RUN),checks=checks,model_reload_max_errors=reload_errors,mbar=mbar_results,source_unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in s.HASHES.items()})
(VAL/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');assert all(summary['source_unchanged'].values())
print('VALIDATION_FINISHED',flush=True)
