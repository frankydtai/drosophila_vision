from pathlib import Path
import sys,json,time,hashlib,os,gc,math
O=Path(__file__).resolve().parent
C=O/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)]
sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import torch
if 'deterministic' in sys.argv[-1]:
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
 torch.use_deterministic_algorithms(True)
import train
import train.cost as cost_module
from train.cost import calc_cost,backward_part_sums
from train.implementation import load_best_node_vals,session_z_from_node_vals
from train.param import schema_clamps,schema_params
from neuron import point_hh

torch.set_num_threads(1)
idx=int(sys.argv[1]);cases=json.loads((O/'cases.json').read_text());case=cases[idx]
device=sys.argv[2] if len(sys.argv)>2 else 'cpu'
limit=int(sys.argv[3]) if len(sys.argv)>3 else case['iterations']
label=sys.argv[4] if len(sys.argv)>4 else 'screen'
out=O/f'case_{idx:02d}_{label}_{device}.json'
opts=json.loads((O/'inputs'/case['run']/'isolated_opts.json').read_text());opts['device']=device
if label.startswith('seq'):opts['sequential']=True
session=train.resolve_session(opts)
if 'pair' in label:
 def pair_cost_bs(current):
  for pack in current.iter_packs():
   bs=cost_module._pack_active_bs(pack,current)
   for start in range(0,len(bs),2):
    active=cost_module._active_cost_pack(pack,current,bs=tuple(bs[start:start+2]))
    if active is not None:yield active,None
 cost_module._iter_cost_bs=pair_cost_bs
session,z0=session_z_from_node_vals(*load_best_node_vals(str(O/'inputs'/case['run'])),session)
assert session.delta_ms==session.delta_ms_pre==1.0
assert opts['fp']==32 and opts['pre_grad'] is True
for name,mod in list(sys.modules.items()):
 if name in ['config','import_bootstrap'] or name.split('.')[0] in ['train','neuron','network','task']:
  path=getattr(mod,'__file__',None)
  if path: assert Path(path).is_relative_to(C),(name,path)
z=torch.nn.Parameter(z0.detach().clone());opt=torch.optim.Adam([z],lr=case['lr'])
clamps=schema_clamps(session.schema,session.sim_dtype).to(device)
forward_seconds=0.0
original_forward=cost_module.forward_trace
def timed_forward(*args,**kwargs):
 global forward_seconds
 if device=="cuda":torch.cuda.synchronize()
 start=time.monotonic();value=original_forward(*args,**kwargs)
 if device=="cuda":torch.cuda.synchronize()
 forward_seconds+=time.monotonic()-start
 return value
cost_module.forward_trace=timed_forward
train_started=time.monotonic()
records=[];reason='completed';started=time.monotonic();sample_stats={}
original_step=point_hh.step
step_counter=0
def watch(grad):
 maximum=float(grad.detach().abs().max())
 if not math.isfinite(maximum):sample_stats['nonfinite_state_gradient']=True
 elif maximum>sample_stats.get('sampled_state_grad_max',0.):sample_stats['sampled_state_grad_max']=maximum
 return grad
def observed_step(*args,**kwargs):
 global step_counter
 state=original_step(*args,**kwargs);step_counter+=1
 if torch.is_grad_enabled() and step_counter%50==0:
  for key in ['v','s']:
   if state[key].requires_grad:state[key].register_hook(watch)
 return state
point_hh.step=observed_step
print('START',idx,case,device,limit,'n_z',len(z),'sequential',session.sequential,flush=True)
source_hashes=json.loads((O/'source_hashes.json').read_text())
def persist():
 payload=dict(temporary_readout_fix=True,microbatch=2 if 'pair' in label else (1 if session.sequential else 'all'),hardware=torch.cuda.get_device_name(0) if device=='cuda' else os.uname().nodename,sequential=session.sequential,case=case,case_index=idx,device=device,limit=limit,label=label,seed='saved_best_params',integrator_step_changed=False,initial_z_sha256=hashlib.sha256(z0.detach().cpu().numpy().tobytes()).hexdigest(),iterations=records,stop_reason=reason,elapsed_s=time.monotonic()-started,source_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()))
 temp=out.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temp.replace(out)
def measure_pair(param_z):
 total=0.0
 with torch.no_grad():
  norm=cost_module._session_part_scale_sum(session)
  for pack,b in cost_module._iter_cost_bs(session):
   params=train.params_from_z(param_z,session)
   parts=cost_module._pack_cost_parts_from_params(params,pack,session,b=b)
   total+=sum(cost_module._part_scale(session,k)/norm*float(v) for k,v in parts.items())
 return total
for it in range(limit):
 tick=time.monotonic();forward_seconds=0.0;sample_stats={};step_counter=0;opt.zero_grad(set_to_none=True)
 rec=dict(iteration=it)
 try:
  if session.sequential:
   cost,_=backward_part_sums(z,session)
  else:
   value=calc_cost(z,session);cost=float(value.detach());value.backward();del value
  grad=z.grad
  rec.update(cost=cost if math.isfinite(cost) else None,grad_max=float(grad.detach().abs().max()) if grad is not None and bool(torch.isfinite(grad).all()) else None,grad_norm64=float(grad.detach().double().norm()) if grad is not None and bool(torch.isfinite(grad).all()) else None,z_finite=bool(torch.isfinite(z).all()),grad_finite=grad is not None and bool(torch.isfinite(grad).all()),**sample_stats)
  if not math.isfinite(cost):reason='nonfinite_cost'
  elif not rec['z_finite']:reason='nonfinite_z'
  elif not rec['grad_finite']:reason='nonfinite_grad'
  elif sample_stats.get('nonfinite_state_gradient'):reason='nonfinite_state_gradient'
  if reason=='completed':
   old_z=z.detach().clone() if 'backtrack' in label else None
   opt.step()
   with torch.no_grad():z.clamp_(clamps[:,0],clamps[:,1])
   adam=opt.state[z]
   rec['adam_finite']=all(bool(torch.isfinite(adam[k]).all()) for k in ['exp_avg','exp_avg_sq'])
   rec['adam_v_max']=float(adam['exp_avg_sq'].max()) if bool(torch.isfinite(adam['exp_avg_sq']).all()) else None
   if not rec['adam_finite']:reason='nonfinite_adam'
   if 'backtrack' in label and reason=='completed':
    delta=z.detach()-old_z;trial_lr=opt.param_groups[0]['lr'];accepted=False;trials=[]
    for attempt in range(4):
     if trial_lr<.0003:break
     factor=trial_lr/opt.param_groups[0]['lr']
     with torch.no_grad():z.copy_((old_z+factor*delta).clamp(clamps[:,0],clamps[:,1]))
     trial_cost=measure_pair(z);trials.append({'lr':trial_lr,'cost':trial_cost if math.isfinite(trial_cost) else None})
     if math.isfinite(trial_cost) and trial_cost<cost-max(1.,cost*1e-4):
      accepted=True;break
     trial_lr*=.5
    rec['trials']=trials;rec['accepted']=accepted
    if accepted:
     rec['accepted_cost']=trial_cost;rec['accepted_lr']=trial_lr
     opt.param_groups[0]['lr']=min(case['lr'],trial_lr*1.5)
    else:
     with torch.no_grad():z.copy_(old_z)
     reason='no_improving_step_above_lr_floor'

 except RuntimeError as exc:
  rec['error']=str(exc);reason='runtime_error'
 rec['forward_s']=forward_seconds;rec['train_elapsed_s']=time.monotonic()-train_started;rec['iteration_s']=time.monotonic()-tick;records.append(rec);persist();print(json.dumps(rec),flush=True)
 if reason!='completed':break
 gc.collect()
point_hh.step=original_step
persist()
print('DONE',idx,reason,'source_unchanged',json.loads(out.read_text())['source_files_unchanged'],flush=True)
