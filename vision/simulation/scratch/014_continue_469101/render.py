from pathlib import Path
import sys,json,hashlib,time
O=Path(__file__).resolve().parent;C=O.parent/'012_adam_resume_diagnosis/copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.implementation as impl
import figure.plot as plot
import task.mbar.pack as packmod
from PIL import Image

torch.set_num_threads(1);build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
selection=json.loads((O/'selected.json').read_text());run=Path(selection['run']);started=time.monotonic()
opts=json.loads((run/'data/train_opts.json').read_text());opts['device']='cpu';session=train.resolve_session(opts)
session,z=impl.session_z_from_node_vals(*impl.load_best_node_vals(str(run)),session)
raw=np.load(run/'data/train_point_hh.npy').reshape(-1);assert np.array_equal(z.detach().numpy(),raw)
with torch.no_grad():verified=float(train.calc_cost(z,session))
final,costs,by_part,final_parts=impl.load_stored_costs(str(run));assert verified==float(final[0])==selection['cost']
assert verified<469100.78125
node_m,node_v,adam_iter,_,_=impl.load_best_adam(str(run));assert adam_iter==selection['adam_iter']
from train.param import z_adams_from_node_vals
loaded_m,loaded_v=z_adams_from_node_vals(node_m,node_v,session.schema,dtype=session.sim_dtype,device='cpu')
with np.load(run.parent/'best_state.npz') as state:
 assert np.array_equal(raw,state['z'])
 assert np.array_equal(loaded_m.numpy(),state['exp_avg']) and np.array_equal(loaded_v.numpy(),state['exp_avg_sq'])
 assert int(state['iter'])==adam_iter
required=['best_param.npz','best_adam.npz','train_point_hh.npy','train_opts.json','costs.npy','best_costs.npy','costs_by_part.npz','best_costs_by_part.npz','point_hh_context.json','point_hh_type_params.csv','point_hh_edge_params.csv']
assert all((run/'data'/name).is_file() for name in required)
assert (run/'param.csv').is_file() and (run/'point_hh_synapse_params.csv').is_file()
plot.plot_cost(costs,str(run/'cost.png'),costs_by_part=by_part,part_order=list(train.session_cost_part_keys(session)),session=session)
plot.plot_rf_t(np.array([raw]),str(run),session=session,final_costs=final,figure_tasks=['spread','mbar','sbar'],model_all=True,plot_right_only=False)
expected=['cost.png']+[f'{task}_{kind}_v.png' for task in ['spread','mbar','sbar'] for kind in ['gt','all']];images={}
for name in expected:
 p=run/name;assert p.is_file() and p.stat().st_size>1000
 with Image.open(p) as im:im.verify()
 with Image.open(p) as im:images[name]=dict(width=im.width,height=im.height,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
protected=json.loads((O/'protected_hashes.json').read_text());changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
result=dict(selected_run=str(run),selected_cost=verified,selected_adam_iter=adam_iter,parameter_roundtrip_exact=True,adam_moments_roundtrip_exact=True,images=images,protected_files_unchanged=True,elapsed_s=time.monotonic()-started)
(O/'render_result.json').write_text(json.dumps(result,indent=2)+'\n');print('VALIDATED',json.dumps(result),flush=True)
