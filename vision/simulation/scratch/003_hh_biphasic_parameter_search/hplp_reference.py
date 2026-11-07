from pathlib import Path
from types import SimpleNamespace
import sys,os,json
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron.forward import forward_v
run=next((SIM/'0_runs/hp_lp').glob('34381872-*'));opts=json.loads((run/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for k in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[k]['delta_ms']=1.;opts[k]['delta_ms_pre']=1.
torch.set_num_threads(1);ss=train.resolve_session(opts);ss,z=session_z_from_node_vals(*load_best_node_vals(str(run)),ss);p=train.params_from_z(z,ss);pack=ss.primary_pack;w=1000;duration=160;total=w+duration+1500+1
i0=pack.i_sti[:,0];ion=pack.i_sti[:,pack.t_onset+10];drive=i0[:,None,:].expand(1,total,-1).clone();drive[:,w:w+duration]=ion[:,None,:]
with torch.no_grad():
 vp=forward_v(ss,p,drive,pack=SimpleNamespace(t_onset=w));vc=forward_v(ss,p,i0[:,None,:].expand_as(drive),pack=SimpleNamespace(t_onset=w))
assert torch.isfinite(vp).all() and torch.isfinite(vc).all();assert torch.equal(vp[:,:w],vc[:,:w]);nc=ss.connectome.node_cells;cells=list(ss.connectome.cells)
a=np.stack([vp[0,:,nc==i].mean(-1).numpy() for i in range(len(cells))],axis=1);b=np.stack([vc[0,:,nc==i].mean(-1).numpy() for i in range(len(cells))],axis=1)
np.savez_compressed(OUT/'validation/HPLP_standard.npz',time_ms=np.arange(total)-w,cells=np.array(cells),pulse=a,control=b,evoked=a-b)
print('FINISHED',flush=True)
