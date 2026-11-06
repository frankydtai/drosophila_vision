from pathlib import Path
import sys, json, time, gc, hashlib
import torch
ROOT=Path('/home/yitai/drosophila_vision/vision/simulation')
sys.path[:0]=[str(ROOT), str(ROOT.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals, session_z_from_node_vals
from train.param import schema_clamps, schema_params
from train.cost import calc_cost
from neuron import point_hh

torch.set_num_threads(1)
run_id=sys.argv[1]
variant=sys.argv[2] if len(sys.argv)>2 else 'baseline'
run=next((ROOT/'0_runs/point_hh').glob(run_id+'-*'))
opts=json.loads((run/'data/train_opts.json').read_text())
opts['device']='cpu'
if variant=='fp64': opts['fp']=64
if variant=='no_pre_grad': opts['pre_grad']=False
session=train.resolve_session(opts)
session,z0=session_z_from_node_vals(*load_best_node_vals(str(run)),session)
if len(sys.argv)>3:
    z0=torch.load(sys.argv[3],map_location='cpu',weights_only=True).to(z0)
z=torch.nn.Parameter(z0.detach().clone())
opt=torch.optim.Adam([z],lr=.1)
clamps=schema_clamps(session.schema,session.sim_dtype)
print('START',run_id,variant,'nodes',session.connectome.n_node,'z',len(z),'packs',[(p.task,tuple(p.i_sti.shape)) for p in session.iter_packs()],flush=True)
source_files=[ROOT/'1_neuron/1_4_point_hh.py',ROOT/'1_neuron/4_forward.py',ROOT/'4_train/3_cost.py',ROOT/'conf/config.yaml']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
records=[]
for iteration in range(1 if len(sys.argv)>3 else 3):
    started=time.monotonic()
    opt.zero_grad()
    counts={'step':0,'nonfinite':False}
    samples=[]
    def hook_for(key,t):
        def hook(grad):
            finite=torch.isfinite(grad)
            maximum=float(grad.abs().max())
            if not bool(finite.all()) and not counts['nonfinite']:
                counts['nonfinite']=True
                bad=(~finite).nonzero()[0].tolist()
                info=dict(event='FIRST_BAD_STATE_GRAD',key=key,t=t,index=bad,max_abs=maximum,nan=int(torch.isnan(grad).sum()),inf=int(torch.isinf(grad).sum()))
                idx=bad[-1]
                if key=='s':
                    info['pre']=str(session.connectome.edge_source_cells[idx]); info['post']=str(session.connectome.edge_target_cells[idx])
                else: info['cell']=str(session.connectome.cells[int(session.connectome.node_cells[idx])])
                print(json.dumps(info),flush=True)
            if key=='v' and t%100==0:
                samples.append([t,maximum])
        return hook
    original_step=point_hh.step
    def observed_step(*args,**kwargs):
        out=original_step(*args,**kwargs)
        counts['step']+=1
        if torch.is_grad_enabled():
            for key,value in out.items():
                if value.requires_grad: value.register_hook(hook_for(key,counts['step']))
        return out
    point_hh.step=observed_step
    try:
        cost=calc_cost(z,session)
        print('FORWARD',iteration,float(cost),'seconds',time.monotonic()-started,flush=True)
        cost.backward()
    finally:
        point_hh.step=original_step
    grad=z.grad.detach()
    bad_params=[]
    for name,spec,start,stop in schema_params(session.schema):
        if stop>start and not bool(torch.isfinite(grad[start:stop]).all()): bad_params.append(name)
    rec=dict(iteration=iteration,cost=float(cost),grad_max=float(grad.abs().max()),bad_params=bad_params,state_v_grad_samples=samples,elapsed=time.monotonic()-started)
    print(json.dumps(rec),flush=True)
    records.append(rec)
    target=Path(__file__).resolve().parent/(run_id+'_'+variant+('_fixed_bad' if len(sys.argv)>3 else '')+'_results.json')
    target.write_text(json.dumps(records,indent=2)+'\n')
    if bad_params:
        torch.save(z.detach(),Path(__file__).resolve().parent/(run_id+'_'+variant+'_bad_z.pt'))
        del cost
        gc.collect()
        print('ANOMALY RERUN',flush=True)
        opt.zero_grad()
        try:
            with torch.autograd.detect_anomaly():
                cost=calc_cost(z,session)
                cost.backward()
        except RuntimeError as exc:
            print('ANOMALY',repr(exc),flush=True)
        break
    opt.step()
    with torch.no_grad(): z.clamp_(clamps[:,0],clamps[:,1])
    del cost
    gc.collect()
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'SOURCE CHANGED'
print('DONE SOURCE UNCHANGED',flush=True)
