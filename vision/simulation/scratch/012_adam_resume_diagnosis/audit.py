from pathlib import Path
import sys,json,numpy as np,torch
O=Path(__file__).resolve().parent; C=O/'copy/vision'
sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import train
from train.implementation import load_init_z
from train.param import schema_params
opts=json.loads((O/'inputs/34432972/data/train_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts);session,z,adam=load_init_z(str(O/'inputs/34431163'),session)
original=np.load(O/'inputs/34431163/data/train_point_hh.npy').reshape(-1)
diff=z.detach().numpy().astype(float)-original
labels=[]
for name,spec,start,stop in schema_params(session.schema):labels.extend([name]*int(stop-start))
report=dict(z_length=len(z),original_z_dtype=str(original.dtype),loaded_z_dtype=str(z.dtype),z_max_abs_roundtrip_error=float(np.max(np.abs(diff))),z_changed_coordinates=int(np.count_nonzero(diff)),z_differences=[dict(index=i,param=labels[i],original=float(original[i]),loaded=float(z[i]),difference=float(diff[i])) for i in range(len(z)) if diff[i]!=0],source_best_adam_iter=adam['iter'])
# Independent synthetic-gradient continuity check for snapshot/load semantics.
import train.optimization as optmod
a=torch.nn.Parameter(torch.arange(5,dtype=torch.float32));oa=torch.optim.Adam([a],lr=.003)
for i in range(5):a.grad=torch.arange(1,6,dtype=torch.float32)*(i+1);oa.step()
b=torch.nn.Parameter(a.detach().clone());ob=torch.optim.Adam([b],lr=.003);optmod._load_adams(ob,b,optmod._adam_bag(oa,a))
for i in range(5,10):
 g=torch.arange(1,6,dtype=torch.float32)*(i+1);a.grad=g.clone();b.grad=g.clone();oa.step();ob.step()
report['synthetic_uninterrupted_vs_resumed_exact']=bool(torch.equal(a,b) and all(torch.equal(oa.state[a][k],ob.state[b][k]) for k in ['step','exp_avg','exp_avg_sq']))
(O/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
