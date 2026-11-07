from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[3]
selection=json.loads((O/'selected.json').read_text());valid=json.loads((O/'render_result.json').read_text());stage=Path(selection['run']);base=json.loads((O/'source_metrics.json').read_text())
assert valid['selected_cost']==selection['cost']<base['total']
assert valid['parameter_roundtrip_exact'] and valid['adam_moments_roundtrip_exact'] and len(valid['images'])==7
assert selection['response_improved']
for key,b in base['groups'].items():
 a=selection['metrics']['groups'][key];assert a['nd_peak_mV']<b['nd_peak_mV']
for key,b in base['parts'].items():
 if key.endswith('_ND'):assert selection['metrics']['parts'][key]<b
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
opts=json.loads((stage/'data/train_opts.json').read_text());assert opts['point_hh']['g_context_mode']=='init' and opts['part_cost_scales']=={'mbar':1.,'mbar_DSI':0.}
report=dict(before=base,after=selection['metrics'],selection_criterion='both ND peaks, ND losses and original total decrease; prefer both ND/PD ratios improving when available; then minimize summed ND loss',selection_tradeoff=selection['selection_tradeoff'],original_loss_weights_unchanged=True)
(stage/'data/nd_optimization_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
provenance=json.loads((stage/'data/provenance.json').read_text());job=provenance['job']
target=ROOT/'vision/simulation/0_runs/point_hh'/f'{job}-init_nd_optimization-loss-{selection["cost"]:.0f}'
assert not target.exists();shutil.copytree(stage,target)
iterations={p.parent.name:len(json.loads(p.read_text())['records']) for p in O.glob('*/progress.json')}
provenance.update(training_origin='continuation_of_native_init_spread100',previous_best=json.loads((O/'campaign_state.json').read_text())['source'],g_context_mode='init',objective_weights_unchanged=True,selection=report['selection_criterion'],ND_before_after=report,optimization_records=str(O),iterations_by_case=iterations,total_additional_iterations=sum(iterations.values()))
(target/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
manifest={}
for p in stage.rglob('*'):
 if p.is_file() and p.relative_to(stage)!=Path('data/provenance.json'):
  q=target/p.relative_to(stage);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==sha;manifest[str(q.relative_to(target))]=sha
images=list(valid['images'])+['nd_comparison.png']
for name in images:
 with Image.open(target/name) as im:im.verify()
assert (target/'nd_comparison.pdf').is_file()
assert float(np.load(target/'data/costs.npy')[0])==selection['cost']
assert int(np.load(target/'data/best_adam.npz')['iter'])==selection['adam_iter']
result=dict(folder=str(target),cost=selection['cost'],previous_cost=base['total'],improvement_percent=100*(1-selection['cost']/base['total']),nd_loss_before=sum(v for k,v in base['parts'].items() if k.endswith('_ND')),nd_loss_after=selection['nd_sum'],selection_tradeoff=selection['selection_tradeoff'],ND_before=base['groups'],ND_after=selection['metrics']['groups'],images=images,parameters_and_adam_verified=True,protected_files_unchanged=True,total_additional_iterations=sum(iterations.values()),files=manifest)
(O/'export_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2),flush=True)
