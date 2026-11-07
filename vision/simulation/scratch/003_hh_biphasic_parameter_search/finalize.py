from pathlib import Path
import json,csv,hashlib,sys,os
import numpy as np
import torch
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1];sys.dont_write_bytecode=True;sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
from figure.plot import load_best
from train.param import params_from_z
import task.mbar.pack as mp
original=mp.build_mbar_signals
mp.build_mbar_signals=lambda *a,**kw:original(*a,**dict(kw,use_cache=False))
torch.set_num_threads(1)
summary=json.loads((OUT/'validation/summary.json').read_text());physics={p:ok for p,ok in summary['source_unchanged'].items() if not p.endswith('/conf/config.yaml')}
assert all(physics.values())
summary['physics_and_trained_source_files_unchanged']=True
summary['main_config_changed_externally_during_validation']=not summary['source_unchanged'][str(SIM/'conf/config.yaml')]
summary['configuration_note']='Saved run options/parameters were loaded once per process. The on-disk main config changed in other ongoing work; no experiment writes to it. Neither HH core file nor the source best_param/train_opts changed.'
summary['validation_complete']=True
api=[]
for gain in [3,6]:
 run=OUT/f'candidate_gain{gain}';ss,z,cost=load_best(str(run));p=params_from_z(z,ss);pairs=list(np.load(run/'data/best_param.npz',allow_pickle=True)['pairs']);assert ss.delta_ms==1. and ss.delta_ms_pre==1.;assert torch.all(p['k_minus_syn']==.01)
 src=ss.connectome.conn.source_idxs;tar=ss.connectome.conn.target_idxs;nc=ss.connectome.node_cells;cells=list(ss.connectome.cells);report=[]
 for target in ['Mi1','Tm3']:
  ix=(nc[src]==cells.index('L1'))&(nc[tar]==cells.index(target));assert torch.all(p['v_th_syn'][ix]==-51.) and torch.all(p['delta_syn'][ix]==.5)
  values=p['gS'][ix]/ss.connectome.edge_n_syn[ix];report.append(dict(pair='L1:'+target,gS_multiplier=gain,v_th_syn_mV=-51.,delta_syn_mV=.5,gS_per_contact_median_nS=float(values.median())))
 info=dict(candidate=str(run),loaded_by_original_load_best=True,delta_ms=1.,k_minus_syn=.01,gene_mode=ss.train_opts['point_hh'].get('gene_mode'),changes=report,evaluated_mbar_cost=cost)
 api.append(info);(run/'native_loader_verification.json').write_text(json.dumps(info,indent=2)+'\n')
summary['original_load_best_verified']=api
(OUT/'validation/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=['search_results.json','balanced_results.json','membrane_results.json','slow_recovery_results.json','slow_balanced_results.json','strongK_results.json','release_shape_results.json','selective_release_results.json','refined_release_results.json']
counts={name:len(json.loads((OUT/name).read_text())) for name in files}
report=dict(recommended_candidate=str(OUT/'candidate_gain3'),stronger_pulse_candidate=str(OUT/'candidate_gain6'),baseline_trained_run=summary['source_run'],dt_ms=1.,k_minus_syn=.01,changed_pairs=['L1:Mi1','L1:Tm3'],recommended_gS_multiplier=3.,v_th_syn_mV=-51.,delta_syn_mV=.5,native_HH_gate_equations_unchanged=True,core_parameter_grid_cases_completed=sum(counts.values()),grid_case_counts=counts,validation_complete=True,formal_source_or_main_config_written_by_experiment=False,limitations=['Bright-pulse biphasic waveforms improved for Mi1/Tm3/T4a; L1 rebound weakened.','Moving-bar reverse peaks remain much smaller than pulse reverse peaks.','Gain3 reduced the recomputed moving-bar cost relative to fixed-k=.01 baseline; gain6 increased it.','This is an evaluated candidate, not a retrained optimum or full HPLP reproduction.'])
(OUT/'result_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
