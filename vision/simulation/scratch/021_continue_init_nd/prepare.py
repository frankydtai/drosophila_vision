from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/yitai/drosophila_vision');S=R/'vision/simulation/scratch';old=S/'020_init_from_scratch';O=S/'021_continue_init_nd';O.mkdir(exist_ok=False)
shutil.copytree(old/'copy',O/'copy',symlinks=True)
protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in json.loads((old/'protected_hashes.json').read_text())}
(O/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n');(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
source=json.loads((old/'export_result.json').read_text())['folder'];common=dict(source=source,native=False,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.},n_iter=40,backtrack=False)
cases=[dict(common,label='nd_resume_0003',resume_adam=True,lr=.0003),dict(common,label='nd_resume_001',resume_adam=True,lr=.001),dict(common,label='nd_reset_0003',resume_adam=False,lr=.0003)]
(O/'screen_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
s=(old/'worker.py').read_text()
s=s.replace("started=time.monotonic();records=[]", "source_total=float(np.load(source/'data/costs.npy')[0])\nwith np.load(source/'data/costs_by_part.npz') as a:\n source_parts={k:float(a[k][0]) for k in a.files}\nnd_keys=[k for k in source_parts if k.endswith('_ND')];assert len(nd_keys)==2\nsource_nd=sum(source_parts[k] for k in nd_keys);best_nd=float('inf');unconstrained_best=float('inf')\nstarted=time.monotonic();records=[]")
s=s.replace('global best_cost,best_z,best_state,best_parts,best_at\n if np.isfinite(value) and value<best_cost:', "global best_cost,best_z,best_state,best_parts,best_at,best_nd,unconstrained_best\n unconstrained_best=min(unconstrained_best,value) if np.isfinite(value) else unconstrained_best\n nd=sum(parts[k] for k in nd_keys)\n eligible=np.isfinite(value) and value<=source_total and all(parts[k]<=source_parts[k] for k in nd_keys)\n if eligible and (nd,value)<(best_nd,best_cost):\n  best_nd=nd")
s=s.replace("report.update(best_cost=best_cost,best_at=best_at", "report.update(best_cost=best_cost,best_nd=best_nd,best_parts=best_parts,source_total=source_total,source_nd=source_nd,source_parts=source_parts,unconstrained_best_cost=unconstrained_best,best_at=best_at")
s=s.replace("rec=dict(iter=it,cost=c,grad_norm=", "rec=dict(iter=it,cost=c,parts=floats,nd_sum=sum(floats[k] for k in nd_keys),grad_norm=")
s=s.replace("selection='lowest_measured_loss'", "selection='lowest_ND_sum_with_both_ND_parts_and_total_at_most_source',selected_nd=best_nd,source_total=source_total,source_nd=source_nd")
(O/'worker.py').write_text(s)
slurm=(old/'extend.slurm').read_text().replace('020_init_from_scratch','021_continue_init_nd').replace('hh_init_extend','hh_init_nd').replace('extend-%A_%a.log','screen-%A_%a.log').replace('worker.py extend_cases.json','worker.py screen_cases.json')
(O/'screen.slurm').write_text(slurm)
state=dict(source=source,records=str(O),g_context_mode='init',phase='ND_screen',selection='both individual ND losses and original total must decrease; select smallest ND loss sum',original_weights_unchanged=True,code_snapshot_from=str(old/'copy'),formal_changes_since_previous_snapshot=[p for p,h in json.loads((old/'protected_hashes.json').read_text()).items() if protected[p]!=h])
(O/'campaign_state.json').write_text(json.dumps(state,indent=2)+'\n')
import ast
ast.parse(s);print(json.dumps(state,indent=2))
