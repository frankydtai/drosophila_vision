from pathlib import Path
import shutil,json,hashlib,subprocess,ast
O=Path(__file__).resolve().parent;P=O.parent/'022_rule_init_nd_comparison';R=O.parents[3]
shutil.copytree(P/'copy',O/'copy',symlinks=True)
for name in ['response_metrics.py','parameter_report.py']:shutil.copy2(P/name,O/name)
protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in json.loads((P/'protected_hashes.json').read_text())};(O/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n');(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
run=str(R/'vision/simulation/0_runs/point_hh/34466728-init_nd_optimization-loss-994279');cases=[dict(label=f'upper_only_{f}',upper_fold=f,kind='upper_only',run=run) for f in [6,8,10,12,16]];cases.append(dict(label='symmetric_12',upper_fold=12,kind='symmetric',run=run));(O/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
s=(P/'compare.slurm').read_text().replace('022_rule_init_nd_comparison','024_mi4_gs_upper_bound').replace('hh_rule_init_cmp','hh_mi4_gs_cap').replace('--array=0-7','--array=0-5');(O/'compare.slurm').write_text(s)
print('Prepared six no-training upper-bound comparisons; connectivity and original loss retained.')
