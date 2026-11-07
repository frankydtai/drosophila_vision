from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[3]
selection=json.loads((O/'selected.json').read_text());valid=json.loads((O/'render_result.json').read_text());stage=Path(selection['run'])
baseline=json.loads((O/'mbar_baseline.json').read_text())['initial_cost']
assert valid['selected_cost']==selection['cost']<baseline
assert valid['parameter_roundtrip_exact'] and valid['adam_moments_roundtrip_exact'] and len(valid['images'])==7
spread=json.loads((O/'spread_native_100/progress.json').read_text());assert spread['complete'] and len(spread['records'])==100
opts=json.loads((stage/'data/train_opts.json').read_text());context=json.loads((stage/'data/point_hh_context.json').read_text())
assert opts['point_hh']['g_context_mode']==context['g_context_mode']=='init'
provenance=json.loads((stage/'data/provenance.json').read_text());progress=json.loads((stage.parent/'progress.json').read_text())
target=ROOT/'vision/simulation/0_runs/point_hh'/f'{provenance["job"]}-init_from_scratch-spread100-mbar-loss-{selection["cost"]:.0f}'
assert not target.exists(),target
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
shutil.copytree(stage,target)
iterations={p.parent.name:len(json.loads(p.read_text())['records']) for p in O.glob('*/progress.json')}
provenance.update(training_origin='native_init_from_scratch',g_context_mode='init',native_random_seed=0,spread_training_iterations=100,spread_initial_cost=spread['initial_cost'],spread_best_cost=spread['best_cost'],spread_best_at=spread['best_at'],spread_source_run=str(O/'spread_native_100/run'),adam_reset_at_task_switch=True,initial_mbar_loss=baseline,improvement_percent_from_mbar_start=100*(1-selection['cost']/baseline),continuation_learning_rate=progress['current_lr'],learning_rate_cap=progress['case']['lr'],optimization_records=str(O),iterations_by_case=iterations,total_computed_iterations=sum(iterations.values()))
(target/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
manifest={}
for p in stage.rglob('*'):
 if p.is_file() and p.relative_to(stage)!=Path('data/provenance.json'):
  q=target/p.relative_to(stage);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==sha;manifest[str(q.relative_to(target))]=sha
for name in valid['images']:
 with Image.open(target/name) as im:im.verify()
assert float(np.load(target/'data/costs.npy')[0])==selection['cost']
assert int(np.load(target/'data/best_adam.npz')['iter'])==selection['adam_iter']
result=dict(folder=str(target),g_context_mode='init',spread_iterations=100,spread_best_cost=spread['best_cost'],mbar_initial_cost=baseline,cost=selection['cost'],mbar_improvement_percent=provenance['improvement_percent_from_mbar_start'],images=list(valid['images']),parameters_and_adam_verified=True,protected_files_unchanged=True,total_computed_iterations=sum(iterations.values()),files=manifest)
(O/'export_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
