from pathlib import Path
import sys,json,hashlib,os,time
O=Path(__file__).resolve().parent; PREV=O.parent/'012_adam_resume_diagnosis'; C=PREV/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.implementation as impl
import figure.plot as plot
import task.mbar.pack as packmod
from PIL import Image

torch.set_num_threads(1)
build_gt=packmod.build_mbar_gt
packmod.build_mbar_gt=lambda *a,**kw:build_gt(*a,**dict(kw,use_cache=False))
run=O/'run';started=time.monotonic()
opts=json.loads((run/'data/train_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts)
session,z=impl.session_z_from_node_vals(*impl.load_best_node_vals(str(run)),session)
raw=np.load(run/'data/train_point_hh.npy').reshape(-1)
assert np.array_equal(z.detach().numpy(),raw), 'parameter save/load roundtrip differs'
with torch.no_grad():reloaded_cost=float(train.calc_cost(z,session))
assert reloaded_cost==469100.78125,reloaded_cost
final,costs,by_part,final_parts=impl.load_stored_costs(str(run))
assert float(final[0])==reloaded_cost
plot.plot_cost(costs,str(run/'cost.png'),costs_by_part=by_part,part_order=list(train.session_cost_part_keys(session)),session=session)
plot.plot_rf_t(np.array([raw]),str(run),session=session,final_costs=final,figure_tasks=['spread','mbar','sbar'],model_all=True,plot_right_only=False)
expected=['cost.png']+[f'{task}_{kind}_v.png' for task in ['spread','mbar','sbar'] for kind in ['gt','all']]
images={}
for name in expected:
 p=run/name
 assert p.is_file() and p.stat().st_size>1000,p
 with Image.open(p) as im:
  im.verify()
 with Image.open(p) as im:images[name]=dict(width=im.width,height=im.height,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert len(images)==7
required=['best_param.npz','best_adam.npz','train_point_hh.npy','train_opts.json','costs.npy','best_costs.npy','costs_by_part.npz','best_costs_by_part.npz','point_hh_context.json','point_hh_type_params.csv','point_hh_edge_params.csv']
assert all((run/'data'/name).is_file() for name in required)
assert (run/'param.csv').is_file() and (run/'point_hh_synapse_params.csv').is_file()
assert int(np.load(run/'data/best_adam.npz')['iter'])==20
protected=json.loads((O/'protected_hashes.json').read_text())
changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not changed,changed
result=dict(selected_cost=reloaded_cost,parameter_roundtrip_exact=True,selected_adam_iter=20,images=images,protected_files_unchanged=True,elapsed_s=time.monotonic()-started)
(O/'render_result.json').write_text(json.dumps(result,indent=2)+'\n')
print('ALL_SEVEN_IMAGES_VALID',json.dumps(result),flush=True)
