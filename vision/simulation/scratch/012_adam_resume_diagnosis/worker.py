from pathlib import Path
import os,sys,json,time,hashlib,gc
O=Path(__file__).resolve().parent; C=O/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)]
sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.cost as costmod
import train.optimization as optimod
import task.mbar.pack as packmod
from train.implementation import load_init_z
from train.param import schema_clamps,adams_from_z,z_adams_from_node_vals
idx=int(sys.argv[1]); device=sys.argv[2] if len(sys.argv)>2 else 'cuda'
mode,lr=[('resume',.003),('reset',.003),('resume_small',.0003),('reset_small',.0003)][idx]
torch.set_num_threads(1)
build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
opts=json.loads((O/'inputs/34432972/data/train_opts.json').read_text()); opts['device']=device
session=train.resolve_session(opts)
session,z0,adam=load_init_z(str(O/'inputs/34431163'),session,resume_adam='auto')
assert adam is not None and adam['iter']==79 and z0.numel()==62
assert session.sim_dtype==torch.float32 and session.delta_ms==session.delta_ms_pre==1
clamps=schema_clamps(session.schema,session.sim_dtype).to(device)
result=dict(mode=mode,lr=lr,device=device,hardware=torch.cuda.get_device_name(0) if device=='cuda' else os.uname().nodename,torch_version=torch.__version__,initial_adam_iter=adam['iter'],records=[])
started=time.monotonic()
def persist():
 result['elapsed_s']=time.monotonic()-started
 (O/f'{mode}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
def measure(z):
 with torch.no_grad():return float(costmod.calc_cost(z,session))
def fresh(lr,reset=False):
 z=torch.nn.Parameter(z0.detach().clone());opt=torch.optim.Adam([z],lr=lr)
 if not reset:optimod._load_adams(opt,z,adam)
 return z,opt
if idx==0:
 # Verify serialization order and state restoration independently of the loss.
 m,v=adams_from_z(adam['exp_avg'].cpu().numpy(),adam['exp_avg_sq'].cpu().numpy(),session.schema)
 rm,rv=z_adams_from_node_vals(m,v,session.schema,dtype=z0.dtype,device=device)
 result['adam_roundtrip_exact']=bool(torch.equal(rm,adam['exp_avg']) and torch.equal(rv,adam['exp_avg_sq']))
 result['adam_finite']=all(bool(torch.isfinite(adam[k]).all()) for k in ['exp_avg','exp_avg_sq'])
 result['adam_v_nonnegative']=bool((adam['exp_avg_sq']>=0).all())
 zp,op=fresh(.003); value=costmod.calc_cost(zp,session); initial=float(value.detach()); value.backward(); grad=zp.grad.detach().clone(); del value
 result['initial_cost']=initial;result['gradient_norm']=float(grad.double().norm());result['gradient_finite']=bool(torch.isfinite(grad).all())
 result['repeat_forward_costs']=[measure(z0),measure(z0)]
 result['probes']=[]
 for reset in [False,True]:
  z,opt=fresh(.003,reset);z.grad=grad.clone();opt.step()
  delta=z.detach()-z0
  actual_delta=delta.clone()
  with torch.no_grad():z.clamp_(clamps[:,0],clamps[:,1])
  result['probes'].append(dict(reset=reset,clamped_coordinates=int((actual_delta!=(z.detach()-z0)).sum()),gradient_dot_full_step=float(torch.dot(grad.double(),delta.double())),step_norm=float(delta.double().norm())))
  record=result['probes'][-1];record['trials']=[]
  for factor in [1.,.1,.01,.001,-.01]:
   trial=(z0+factor*delta).clamp(clamps[:,0],clamps[:,1])
   record['trials'].append(dict(lr_equivalent=.003*factor,cost=measure(trial)))
  if not reset:
   mm=.9*adam['exp_avg']+.1*grad; vv=.999*adam['exp_avg_sq']+.001*grad.square();t=adam['iter']+1
   manual=z0-.003*(mm/(1-.9**t))/(torch.sqrt(vv/(1-.999**t))+1e-8)
   result['manual_adam_update_max_error']=float((manual-(z0+delta)).abs().max())
  persist();print('PROBE',json.dumps(record),flush=True)
 del zp,op,grad;gc.collect()
z,opt=fresh(lr,reset=mode.startswith('reset'))
for it in range(10 if idx==3 else 20):
 opt.zero_grad(set_to_none=True);value=costmod.calc_cost(z,session);c=float(value.detach());value.backward();del value
 grad=z.grad
 if not np.isfinite(c) or grad is None or not bool(torch.isfinite(grad).all()):
  result['error']=dict(iter=it,cost=c if np.isfinite(c) else None,nonfinite_grad=True);persist();raise RuntimeError(result['error'])
 old=z.detach().clone();opt.step()
 with torch.no_grad():z.clamp_(clamps[:,0],clamps[:,1])
 rec=dict(iter=it,cost=c,grad_norm=float(grad.double().norm()),gradient_dot_step=float(torch.dot(grad.double(),(z.detach()-old).double())),adam_iter=int(opt.state[z]['step']),step_norm=float((z.detach()-old).double().norm()))
 result['records'].append(rec);persist();print('STEP',json.dumps(rec),flush=True);gc.collect()
result['final_cost']=measure(z)
result['best_cost']=min([result['final_cost']]+[r['cost'] for r in result['records']])
# Verify snapshot files remain identical; production manifest retained for final audit.
result['copy_files_unchanged']=all(hashlib.sha256((O/'copy'/Path(p).relative_to('/home/yitai/drosophila_vision')).read_bytes()).hexdigest()==h for p,h in json.loads((O/'source_hashes.json').read_text()).items() if '/0_runs/' not in p)
persist();print('DONE',mode,result['final_cost'],result['best_cost'],flush=True)
