from pathlib import Path
import sys,json,io,time
O=Path(__file__).resolve().parent;C=O/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.implementation as impl
import train.optimization as optimod
import task.mbar.pack as packmod

torch.set_num_threads(1);build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
spread=O/'spread_native_100';sp=json.loads((spread/'progress.json').read_text());assert sp['complete'] and len(sp['records'])==100
candidates=[]
for case in json.loads((O/'mbar_cases.json').read_text()):
 p=O/case['label'];d=json.loads((p/'progress.json').read_text());candidates.append((d['best_cost'],p,d))
_,parent,progress=min(candidates,key=lambda x:x[0])
state=None
for attempt in range(5):
 try:
  blob=(parent/'best_state.npz').read_bytes()
  with np.load(io.BytesIO(blob)) as d:state={k:d[k].copy() for k in d.files}
  break
 except (OSError,ValueError,EOFError):time.sleep(.2)
assert state is not None
folder=O/'early_refine_seed';folder.mkdir(exist_ok=True);(folder/'best_state.npz').write_bytes(blob)
opts=json.loads((parent/'resolved_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts);session,_=impl.session_z_from_node_vals(*impl.load_best_node_vals(str(spread/'run')),session)
z=torch.tensor(state['z'],dtype=session.sim_dtype,device='cpu')
assert session.model_context.provenance['g_context_mode']=='init'
with torch.no_grad():
 parts=train.calc_cost_parts(z,session);value=float(train.calc_cost(z,session,parts=parts))
assert value==float(state['cost']),(value,float(state['cost']))
result=optimod.TrainResult(run_params=z.numpy()[None,:].astype(np.float64),final_costs=np.array([value]),costs=np.array([r['cost'] for r in progress['records']]),final_costs_by_part={k:np.array([float(v)]) for k,v in parts.items()},run_adams=({'exp_avg':state['exp_avg'].astype(np.float64),'exp_avg_sq':state['exp_avg_sq'].astype(np.float64),'iter':int(state['iter'])},))
impl.save_train_data('train_point_hh.npy',str(folder/'run'),session,result)
metadata=dict(source_case=parent.name,captured_cost=value,captured_adam_iter=int(state['iter']),source_iterations_observed=len(progress['records']),mode='init',native_spread_steps=100,additional_optimizer_iterations=0)
(folder/'run/data/provenance.json').write_text(json.dumps(metadata,indent=2)+'\n');(O/'refine_seed.json').write_text(json.dumps(metadata,indent=2)+'\n')
cases=[dict(label=f'refine_{i:02d}_{"resume" if resume else "reset"}',source=str(folder/'run'),native=False,resume_adam=resume,lr=.0003,n_iter=32,backtrack=True,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.}) for i,resume in enumerate([True,False])]
assert not (O/'refine_cases.json').exists()
(O/'refine_cases.json').write_text(json.dumps(cases,indent=2)+'\n');print('SEED_VALIDATED',json.dumps(metadata),flush=True)
