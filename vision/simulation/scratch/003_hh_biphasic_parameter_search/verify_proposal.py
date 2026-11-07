from pathlib import Path
import json,sys
import torch
from omegaconf import OmegaConf
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
import task.mbar.pack as mp
orig=mp.build_mbar_signals
mp.build_mbar_signals=lambda *a,**kw:orig(*a,**dict(kw,use_cache=False))
torch.set_num_threads(1);cfg=OmegaConf.to_container(OmegaConf.load(OUT/'formal_config_proposal.yaml'),resolve=True);run=OUT/'candidate_gain3';opts=json.loads((run/'data/train_opts.json').read_text());opts['params']=cfg['params'];opts['point_hh']=cfg['point_hh'];opts['point_hh']['synapse_rules']=str(run/'synapse_rules.csv');opts['device']='cpu'
ss=train.resolve_session(opts);ss,z=session_z_from_node_vals(*load_best_node_vals(str(run)),ss);p=train.params_from_z(z,ss);nc=ss.connectome.node_cells;cells=list(ss.connectome.cells);src=ss.connectome.conn.source_idxs;tar=ss.connectome.conn.target_idxs
assert torch.all(p['k_minus_syn']==.01)
for c in ['Mi1','Tm3']:
 mask=(nc[src]==cells.index('L1'))&(nc[tar]==cells.index(c));assert torch.all(p['v_th_syn'][mask]==-51.) and torch.all(p['delta_syn'][mask]==.5)
(OUT/'proposal_validation.json').write_text(json.dumps(dict(valid_yaml=True,valid_native_schema=True,candidate_initialization_applied=True,k_minus_fixed=True,modified_pair_values_verified=True,formal_files_modified=False,rules_path_substitution='Proposed future rules path replaced by identical tested scratch rules during validation only.'),indent=2)+'\n')
print('PROPOSAL_VALIDATED',flush=True)
