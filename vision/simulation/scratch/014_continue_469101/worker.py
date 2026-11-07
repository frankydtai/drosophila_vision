from pathlib import Path
import sys,os,json,time,gc,hashlib,argparse
O=Path(__file__).resolve().parent;C=O.parent/'012_adam_resume_diagnosis/copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.cost as costmod
import train.optimization as optimod
import train.implementation as impl
import task.mbar.pack as packmod
from train.param import schema_clamps
parser=argparse.ArgumentParser();parser.add_argument('cases');parser.add_argument('index',type=int);args=parser.parse_args()
case=json.loads(Path(args.cases).read_text())[args.index];folder=O/case['label'];folder.mkdir(exist_ok=True)
torch.set_num_threads(1)
build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
source=Path(case['source']);opts=json.loads((source/'data/train_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts);session,z0,adam=impl.load_init_z(str(source),session,resume_adam=case['resume_adam'])
z=torch.nn.Parameter(z0.detach().clone());opt=torch.optim.Adam([z],lr=case['lr'])
if adam is not None:optimod._load_adams(opt,z,adam)
clamps=schema_clamps(session.schema,session.sim_dtype).to('cpu')
assert z.numel()==62 and session.sim_dtype==torch.float32 and session.delta_ms==session.delta_ms_pre==1
assert session.train_opts['part_cost_scales']=={'mbar':1.,'mbar_DSI':0.}
started=time.monotonic();records=[];histories={};best_cost=float('inf');best_z=None;best_state=None;best_parts=None;best_at=None;reason='completed'
report=dict(case=case,hostname=os.uname().nodename,torch_version=torch.__version__,records=records)
def measure(current):
 with torch.no_grad():
  p=costmod.calc_cost_parts(current,session);v=float(costmod.calc_cost(current,session,parts=p));return v,{k:float(x) for k,x in p.items()}
def maybe_best(value,parts,at):
 global best_cost,best_z,best_state,best_parts,best_at
 if np.isfinite(value) and value<best_cost:
  best_cost=value;best_z=z.detach().clone();best_state=optimod._adam_bag(opt,z);best_parts=parts.copy();best_at=at
  np.savez(folder/'best_state.npz',z=best_z.numpy(),exp_avg=best_state['exp_avg'].numpy(),exp_avg_sq=best_state['exp_avg_sq'].numpy(),iter=best_state['iter'],cost=value)
def persist():
 report.update(best_cost=best_cost,best_at=best_at,elapsed_s=time.monotonic()-started,stop_reason=reason,current_lr=opt.param_groups[0]['lr'])
 temp=folder/'progress.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(folder/'progress.json')
for it in range(case['n_iter']):
 opt.zero_grad(set_to_none=True)
 try:
  parts=costmod.calc_cost_parts(z,session);value=costmod.calc_cost(z,session,parts=parts);c=float(value.detach());floats={k:float(v.detach()) for k,v in parts.items()};value.backward();del value,parts
 except RuntimeError as exc:reason='runtime_error';report['error']=str(exc);break
 if not np.isfinite(c) or z.grad is None or not bool(torch.isfinite(z.grad).all()):reason='nonfinite_cost_or_gradient';break
 if it==0:report['initial_cost']=c
 maybe_best(c,floats,it)
 for k,v in floats.items():histories.setdefault(k,[]).append(v)
 rec=dict(iter=it,cost=c,grad_norm=float(z.grad.double().norm()),lr=opt.param_groups[0]['lr'])
 if case.get('backtrack'):
  old=z.detach().clone();before=optimod._adam_bag(opt,z);opt.step();delta=z.detach()-old;trials=[];accepted=False
  for factor in [1.,.3,.1,.03,.01]:
   with torch.no_grad():z.copy_((old+factor*delta).clamp(clamps[:,0],clamps[:,1]))
   trial,trial_parts=measure(z);trials.append(dict(factor=factor,cost=trial if np.isfinite(trial) else None))
   if np.isfinite(trial) and trial<c-max(.1,abs(c)*1e-7):
    accepted=True;opt.param_groups[0]['lr']=min(case['lr'],max(1e-7,opt.param_groups[0]['lr']*factor*1.25));maybe_best(trial,trial_parts,it+1);break
  rec.update(trials=trials,accepted=accepted)
  if not accepted:
   with torch.no_grad():z.copy_(old)
   optimod._load_adams(opt,z,before);new_lr=opt.param_groups[0]['lr']*.3
   opt=torch.optim.Adam([z],lr=new_lr);rec['reset_after_rejected_trials']=True
   if opt.param_groups[0]['lr']<1e-7:reason='no_improving_step';records.append(rec);persist();break
 else:
  opt.step()
  with torch.no_grad():z.clamp_(clamps[:,0],clamps[:,1])
 rec['adam_iter']=optimod._adam_bag(opt,z)['iter'];records.append(rec);persist();print('STEP',json.dumps(rec), 'BEST',best_cost,flush=True);gc.collect()
final,fp=measure(z);maybe_best(final,fp,len(records));report['terminal_cost']=final
assert best_z is not None and best_state is not None
with torch.no_grad():
 p=costmod.calc_cost_parts(best_z,session);verified=float(costmod.calc_cost(best_z,session,parts=p))
assert verified==best_cost,(verified,best_cost)
result=optimod.TrainResult(run_params=best_z.numpy()[None,:].astype(np.float64),final_costs=np.array([best_cost]),costs=np.array([r['cost'] for r in records]),costs_by_part={k:np.array(v) for k,v in histories.items()},final_costs_by_part={k:np.array([v]) for k,v in best_parts.items()},run_adams=({'exp_avg':best_state['exp_avg'].numpy().astype(np.float64),'exp_avg_sq':best_state['exp_avg_sq'].numpy().astype(np.float64),'iter':best_state['iter']},))
impl.save_train_data('train_point_hh.npy',str(folder/'run'),session,result)
provenance=dict(source_run=str(source),optimization_case=case,selected_cost=best_cost,selected_update=best_at,selected_adam_iter=best_state['iter'],selection='lowest_measured_loss',terminal_cost=final,job=os.environ.get('SLURM_JOB_ID'),array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),device='cpu',hostname=os.uname().nodename,torch_version=torch.__version__)
(folder/'run/data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
report['selected_adam_iter']=best_state['iter'];report['complete']=True;persist();print('DONE',case['label'],best_cost,reason,flush=True)
