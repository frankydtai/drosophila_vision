from pathlib import Path
import sys,json,time
from dataclasses import replace
O=Path(__file__).resolve().parent;C=O/'copy/vision';sys.path[:0]=[str(C/'simulation'),str(C)];sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import torch,train
from train.cost import calc_cost,backward_part_sums,calc_cost_parts
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron.readout import pack_traces,window_time_traces
from types import SimpleNamespace
torch.set_num_threads(1)
t=torch.arange(20,dtype=torch.float32).reshape(1,20,1)
p=SimpleNamespace(t_onset=10,gts=torch.zeros(1,10),i_sti=t,entry_bs=torch.tensor([0]),entry_nodes=torch.tensor([0]),cost_t0s=torch.tensor([5]),cost_zero_before_t_onset=False)
assert torch.equal(pack_traces(t,p),torch.arange(5,15,dtype=torch.float32)[None])
assert torch.equal(window_time_traces(t,p.entry_bs,p.entry_nodes,p.cost_t0s,10,t_onset=10)[0,:5],torch.zeros(5))
opts=json.loads((O/'inputs/34401153/isolated_opts.json').read_text());opts['device']='cpu'
session=train.resolve_session(opts);session,z0=session_z_from_node_vals(*load_best_node_vals(str(O/'inputs/34401153')),session)
values={};grads={}
for mode in ['full','sequential']:
 current=replace(session,sequential=(mode=='sequential'))
 z=z0.detach().clone().requires_grad_()
 if mode=='full':
  value=calc_cost(z,current);values[mode]=float(value.detach());value.backward();del value
 else:values[mode],_=backward_part_sums(z,current)
 grads[mode]=z.grad.detach().clone()
 print(mode,values[mode],float(grads[mode].abs().max()),flush=True)
cost_rel=abs(values['full']-values['sequential'])/abs(values['full'])
grad_rel=float((grads['full'].double()-grads['sequential'].double()).norm()/grads['full'].double().norm())
result=dict(costs=values,cost_relative_difference=cost_rel,gradient_relative_l2_difference=grad_rel,synthetic_readout_regression=True,cost_pass=cost_rel<1e-5,gradient_pass=grad_rel<1e-4,formal_edits=False)
(O/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
assert result['cost_pass'] and result['gradient_pass']
