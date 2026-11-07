from pathlib import Path
import os,sys,json,copy
O=Path(__file__).resolve().parent
R=O.parents[3]
C=O/'copy/vision'
if os.environ.get('G_CONTEXT_TEST_TARGET')=='formal': C=R/'vision'
sys.path[:0]=[str(C/'simulation'),str(C)]
sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import neuron.point_hh as hh
from train.session import load_train_connectome
import config
from types import SimpleNamespace
assert Path(hh.__file__).resolve()==(C/'simulation/1_neuron/1_4_point_hh.py').resolve()
torch.set_num_threads(1)
run=R/'vision/simulation/0_runs/point_hh/34437673-continued_optimization-loss-282967'
opts=json.loads((run/'data/train_opts.json').read_text())
n=load_train_connectome(opts['network_json'],device='cpu',a_syn_exc=config.MODEL['a_syn_exc'],a_syn_inh=config.MODEL['a_syn_inh'],sim_dtype=torch.float32,syn_mode=opts['syn_mode'])
rule=hh.build_context(n,opts)
source=json.loads((run/'data/point_hh_context.json').read_text())
assert rule.provenance['resolution_counts']==source['resolution_counts']
params=dict(rule.params)
init_opts=copy.deepcopy(opts)
init_opts['point_hh']['g_context_mode']='init'
ctx=hh.build_context(n,init_opts)
for p in ('gNa','gK','gLeak'):
    assert bool((ctx.params[p]==init_opts['params'][p]['init']).all())
torch.testing.assert_close(ctx.params['gS'],n.edge_n_syn*init_opts['params']['gS']['init'],rtol=1e-6,atol=1e-7)
for p in ('cap','eNa','eK','eLeak','e_syn','v_th_syn','delta_syn','k_minus_syn'):
    assert torch.equal(rule.params[p],ctx.params[p])
old_zero=rule.params['gS']==0
assert int(old_zero.sum())==3309
assert bool((ctx.params['gS'][old_zero]>0).all())
assert ctx.provenance['resolution_counts']['unresolved_init']==3309
with np.load(run/'data/best_param.npz',allow_pickle=True) as saved:
    decoded={name:torch.tensor(saved[name],dtype=torch.float32) for name in hh.GENE_PARAM_NAMES}
zero={name:torch.zeros_like(v) for name,v in decoded.items()}
sess=SimpleNamespace(model_context=ctx,train_opts=init_opts,connectome=n)
base=hh.prepare_training_params(zero,sess)
for p in ('gNa','gK','gLeak','gS'): assert torch.equal(base[p],ctx.params[p])
physical=hh.prepare_training_params(decoded,sess)
assert all(bool(torch.isfinite(physical[p]).all()) for p in ('gNa','gK','gLeak','gS'))
phase=os.environ.get('G_CONTEXT_TEST_PHASE','candidate')
hh.save_context(ctx,O/('real_init_'+phase))
hh.save_context(rule,O/('real_rule_'+phase))
report={'phase':phase,'model_source':hh.__file__,'n_cells':n.n_cell,'n_nodes':n.n_node,'n_edges':n.conn.n_edge,'init_values':ctx.provenance['g_inits'],'all_edges_use_init_times_contacts':True,'previously_unresolved_edges_enabled':int(old_zero.sum()),'non_G_parameters_unchanged':True,'zero_gene_weights_recover_selected_context':True,'gene_mapped_conductances_finite':True,'cache_init_change':None}
init_opts['params']['gS']['init']=.7
new=hh.build_context(n,init_opts)
assert new is not ctx
torch.testing.assert_close(new.params['gS'],n.edge_n_syn*.7,rtol=1e-6,atol=1e-7)
report['cache_init_change']=True
(O/('integration_'+phase+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
