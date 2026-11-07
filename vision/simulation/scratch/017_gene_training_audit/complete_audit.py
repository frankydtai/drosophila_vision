from pathlib import Path
import sys, json, hashlib, time
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
COPY = OUT.parent / '012_adam_resume_diagnosis/copy/vision'
RUN = ROOT / 'vision/simulation/0_runs/point_hh/34437673-continued_optimization-loss-282967'
report = json.loads((OUT/'results.json').read_text())
report['checks']['full_objective_backward'] = {'completed': False, 'exit_code': 137, 'reason': 'process killed by system; cause not established'}
paths = []
for dirname in ('1_neuron','2_network','3_task','4_train','5_figure','conf'):
    paths.extend(p for p in (ROOT/'vision/simulation'/dirname).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml','.json'))
before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'formal_hashes_before_completion.json').write_text(json.dumps(before,indent=2)+'\n')
sys.path[:0] = [str(COPY/'simulation'),str(COPY)]
sys.modules.pop('import_bootstrap',None)
import import_bootstrap
import numpy as np
import torch
import train
import train.implementation as impl
import train.param as parammod
import neuron.point_hh as hh
import task.mbar.pack as packmod

torch.set_num_threads(1)
original_build_gt = packmod.build_mbar_gt
packmod.build_mbar_gt = lambda *a,**kw: original_build_gt(*a,**dict(kw,use_cache=False))
opts = json.loads((RUN/'data/train_opts.json').read_text())
opts['device'] = 'cpu'
session = train.resolve_session(opts)
node_vals,cells,pairs = impl.load_best_node_vals(str(RUN))
session,z = impl.session_z_from_node_vals(node_vals,cells,pairs,session)
z = z.detach().clone().requires_grad_(True)
params = train.params_from_z(z,session)
v0 = torch.full((1,session.connectome.n_node),-55.,dtype=session.sim_dtype)
state = hh.initial_state(v0,n_edge=session.connectome.conn.n_edge)
for step in range(12):
    stimulus = torch.zeros_like(v0)
    stimulus[:,session.connectome.node_cells == 0] = 40. if step >= 3 else 0.
    state = hh.step(state,params,stimulus,session,delta_ms=1.)
value = state['v'].square().mean()
value.backward()
assert torch.isfinite(value) and z.grad is not None and torch.isfinite(z.grad).all()
report['checks']['short_HH_path_backward'] = {'passed': True, 'steps': 12, 'n_nodes': session.connectome.n_node, 'n_edges': session.connectome.conn.n_edge, 'loss': float(value.detach()), 'gradient_norm': float(z.grad.double().norm()), 'all_62_gradients_finite': True, 'scope': 'synthetic short stimulus and voltage-square objective; not the full experimental loss gradient'}
gradients = {name: float(z.grad[start]) for name,spec,start,stop in parammod.schema_params(session.schema) if name in hh.GENE_PARAM_NAMES}
(OUT/'short_path_gene_gradients.json').write_text(json.dumps(gradients,indent=2)+'\n')
report['checks']['short_path_gene_gradient_summary'] = {'n_total':len(gradients),'n_nonzero':sum(v!=0 for v in gradients.values())}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
report['checks']['formal_files_unchanged_during_completion'] = {'passed':True,'n_files':len(before)}
report['completed'] = True
report['limitations'] = ['Full long-sequence experimental objective backward was killed with exit code 137 and did not complete. Full objective forward reproduced exactly; all 49 conductance-mapping gradients passed finite differences; short HH path backward passed.']
report['elapsed_seconds'] = time.time()-report['started_at_unix']
(OUT/'results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report['checks'].items() if k.startswith(('short_','formal_files','full_saved','full_objective_backward'))},indent=2),flush=True)
