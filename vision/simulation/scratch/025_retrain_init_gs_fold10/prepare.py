from pathlib import Path
import json,hashlib,shutil,subprocess,difflib,ast
from omegaconf import OmegaConf
O=Path(__file__).resolve().parent;R=O.parents[3];old=O.parent/'024_mi4_gs_upper_bound';p=R/'vision/simulation/conf/config.yaml'
protected_before={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in json.loads((old/'protected_hashes.json').read_text())};(O/'protected_hashes_before.json').write_text(json.dumps(protected_before,indent=2)+'\n');(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
before=p.read_text();line='max_fold_changes: {gNa: 4.0, gK: 3.5, gLeak: 3.0, gS: 6.0}';assert before.count(line)==1;after=before.replace(line,line.replace('gS: 6.0','gS: 10.0'));(O/'config_before.yaml').write_text(before);(O/'config_after.yaml').write_text(after)
a=OmegaConf.to_container(OmegaConf.create(before),resolve=True);b=OmegaConf.to_container(OmegaConf.create(after),resolve=True);assert b['point_hh']['gene']['max_fold_changes']['gS']==10.;b['point_hh']['gene']['max_fold_changes']['gS']=6.;assert a==b
p.write_text(after)
(O/'authorized_config_change.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/vision/simulation/conf/config.yaml',tofile='b/vision/simulation/conf/config.yaml')))
shutil.copytree(old/'copy',O/'copy',symlinks=True);shutil.copy2(p,O/'copy/vision/simulation/conf/config.yaml')
protected_after={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in protected_before};changed=[f for f in protected_before if protected_before[f]!=protected_after[f]];assert changed==[str(p)],changed
for f,h in protected_after.items():
 q=O/'copy'/Path(f).relative_to(R);assert hashlib.sha256(q.read_bytes()).hexdigest()==h,f
(O/'protected_hashes.json').write_text(json.dumps(protected_after,indent=2)+'\n')
source=str(R/'vision/simulation/0_runs/point_hh/34466728-init_nd_optimization-loss-994279')
common=dict(source=source,native=False,resume_adam=False,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.})
cases=[dict(common,label='screen_0003',lr=.0003,n_iter=48,backtrack=False),dict(common,label='screen_0001',lr=.0001,n_iter=48,backtrack=False),dict(common,label='screen_backtrack',lr=.0001,n_iter=32,backtrack=True)]
(O/'screen_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
s=(O.parent/'021_continue_init_nd/worker.py').read_text();s=s.replace("session=train.resolve_session(opts);session,z0,adam=", "opts['point_hh']['gene']['max_fold_changes']['gS']=10.0\n session=train.resolve_session(opts);session,z0,adam=")
s=s.replace("ctx=session.model_context", "assert session.train_opts['point_hh']['gene']['max_fold_changes']['gS']==10.0\nif not case['resume_adam']:assert adam is None\nctx=session.model_context")
s=s.replace(" if eligible and (nd,value)<(best_nd,best_cost):", " if eligible and value<best_cost:")
s=s.replace("selection='lowest_ND_sum_with_both_ND_parts_and_total_at_most_source'", "selection='lowest_original_total_with_both_ND_part_losses_at_most_previous_checkpoint',gS_max_fold=10.0,gS_per_contact_range_nS=[.03,3.],adam_reset_at_fold_change=True")
# Keep the original physical bounds audit independent of optimizer updates.
s=s.replace("initial,initial_parts=measure(z);report['initial_cost']=initial;", "initial,initial_parts=measure(z)\nif case.get('verify_initial'):\n assert initial==855180.6875,(initial,855180.6875)\nreport['initial_cost']=initial;")
for c in cases:c['verify_initial']=True
(O/'screen_cases.json').write_text(json.dumps(cases,indent=2)+'\n');ast.parse(s);(O/'worker.py').write_text(s)
s=(O.parent/'021_continue_init_nd/screen.slurm').read_text().replace('021_continue_init_nd','025_retrain_init_gs_fold10').replace('hh_init_nd','hh_gs10_train');(O/'screen.slurm').write_text(s)
state=dict(phase='training',source=source,g_context_mode='init',gene_mode='on',gS_max_fold_before=6.,gS_max_fold_after=10.,gS_per_contact_range_before=[.05,1.8],gS_per_contact_range_after=[.03,3.],formal_change=str(p),formal_change_scope='one scalar gS max_fold_changes 6 to 10; user approved both upper and lower expansion',old_loss=994279.375,new_mapping_initial_loss=855180.6875,adam_reset=True)
(O/'campaign_state.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state,indent=2))
