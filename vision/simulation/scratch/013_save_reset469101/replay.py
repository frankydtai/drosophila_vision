from pathlib import Path
import sys,os,json,time,gc,hashlib
O=Path(__file__).resolve().parent; ROOT=O.parents[3]; PREV=O.parent/'012_adam_resume_diagnosis'; C=PREV/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.cost as costmod
import train.optimization as optimod
import task.mbar.pack as packmod
from train.implementation import load_init_z,save_train_data
from train.param import schema_clamps

torch.set_num_threads(1)
build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
expected=json.loads((PREV/'reset.json').read_text())
opts=json.loads((PREV/'inputs/34432972/data/train_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts)
session,z0,_=load_init_z(str(PREV/'inputs/34431163'),session,resume_adam='auto')
z=torch.nn.Parameter(z0.detach().clone());opt=torch.optim.Adam([z],lr=.003)
clamps=schema_clamps(session.schema,session.sim_dtype).to('cpu')
assert z.numel()==62 and session.sim_dtype==torch.float32 and session.delta_ms==session.delta_ms_pre==1
records=[];histories={};started=time.monotonic()
for it in range(20):
 opt.zero_grad(set_to_none=True)
 parts=costmod.calc_cost_parts(z,session);value=costmod.calc_cost(z,session,parts=parts);c=float(value.detach())
 floats={k:float(v.detach()) for k,v in parts.items()}
 value.backward();del value,parts
 assert c==expected['records'][it]['cost'],(it,c,expected['records'][it]['cost'])
 assert bool(torch.isfinite(z.grad).all())
 grad_norm=float(z.grad.double().norm())
 assert grad_norm==expected['records'][it]['grad_norm'],(it,grad_norm,expected['records'][it]['grad_norm'])
 for k,v in floats.items():histories.setdefault(k,[]).append(v)
 opt.step()
 with torch.no_grad():z.clamp_(clamps[:,0],clamps[:,1])
 records.append(dict(iter=it,cost=c,grad_norm=grad_norm,adam_iter=int(opt.state[z]['step'])))
 (O/'replay_progress.json').write_text(json.dumps(dict(records=records,elapsed_s=time.monotonic()-started),indent=2)+'\n')
 state=optimod._adam_bag(opt,z)
 np.savez(O/'replay_checkpoint.npz',z=z.detach().numpy(),exp_avg=state['exp_avg'].numpy(),exp_avg_sq=state['exp_avg_sq'].numpy(),iter=state['iter'])
 print('REPLAY',it,c,'EXACT',flush=True);gc.collect()
with torch.no_grad():
 parts=costmod.calc_cost_parts(z,session);final=float(costmod.calc_cost(z,session,parts=parts));final_parts={k:np.array([float(v)]) for k,v in parts.items()}
assert final==expected['final_cost']==469100.78125,(final,expected['final_cost'])
state=optimod._adam_bag(opt,z)
result=optimod.TrainResult(run_params=z.detach().numpy()[None,:].astype(np.float64),final_costs=np.array([final]),costs=np.array([r['cost'] for r in records]),costs_by_part={k:np.array(v) for k,v in histories.items()},final_costs_by_part=final_parts,run_adams=({'exp_avg':state['exp_avg'].numpy().astype(np.float64),'exp_avg_sq':state['exp_avg_sq'].numpy().astype(np.float64),'iter':state['iter']},))
run=O/'run';assert not run.exists(),run
save_train_data('train_point_hh.npy',str(run),session,result)
source=next((ROOT/'vision/simulation/0_runs/point_hh').glob('34431163-*'))
provenance=dict(source_run=str(source),diagnostic_record=str(PREV/'reset.json'),diagnostic_job='34434745_1',replay_job=os.environ.get('SLURM_JOB_ID'),selection='final_after_20_adam_updates',selected_cost=final,trajectory_best_cost=expected['best_cost'],selected_adam_iter=20,resume_adam=False,learning_rate=.003,n_iter=20,device='cpu',hardware=os.uname().nodename,torch_version=torch.__version__,all_20_costs_and_gradient_norms_match_diagnostic_exactly=True,elapsed_s=time.monotonic()-started)
(run/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
(O/'replay_result.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('SAVED_EXACT_FINAL',final,run,flush=True)
