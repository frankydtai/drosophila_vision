from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[3];s=json.loads((O/'selected.json').read_text());valid=json.loads((O/'render_result.json').read_text());stage=Path(s['run']);old=json.loads((O/'old_mapping_metrics.json').read_text());new=json.loads((O/'new_mapping_metrics.json').read_text())
assert valid['selected_cost']==s['cost']<=new['total']<old['total'];assert valid['parameter_roundtrip_exact'] and valid['adam_moments_roundtrip_exact'];assert s['ND_response_improved'] and s['gS_range_valid']
opts=json.loads((stage/'data/train_opts.json').read_text());context=json.loads((stage/'data/point_hh_context.json').read_text());assert opts['point_hh']['gene']['max_fold_changes']['gS']==context['gene_mapping']['max_fold_changes']['gS']==10.
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());before=json.loads((O/'protected_hashes_before.json').read_text());changed=[p for p in before if before[p]!=protected[p]];assert changed==[str(ROOT/'vision/simulation/conf/config.yaml')]
report=dict(old_mapping=old,new_mapping_before_training=new,retrained=s['metrics'],new_per_contact_range_nS=[.03,3.],old_per_contact_range_nS=[.05,1.8],original_MSE_definition_and_weights_unchanged=True,connectivity_unchanged=True)
for name in ['effective_membrane_conductances.csv','effective_synapse_conductances.csv']:
 shutil.copy2(stage.parent/name,stage/'data'/name)
(stage/'data/nd_optimization_metrics.json').write_text(json.dumps(report,indent=2)+'\n');provenance=json.loads((stage/'data/provenance.json').read_text());target=ROOT/'vision/simulation/0_runs/point_hh'/f'{provenance["job"]}-init_gSfold10-retrained-loss-{s["cost"]:.0f}';assert not target.exists();shutil.copytree(stage,target)
iterations={p.parent.name:len(json.loads(p.read_text())['records']) for p in O.glob('*/progress.json')};provenance.update(training_origin='continuation_of_native_init_spread100',original_best=json.loads((O/'campaign_state.json').read_text())['source'],gS_max_fold=10.,gS_per_contact_range_nS=[.03,3.],adam_reset_when_changing_fold=True,authorized_formal_change='point_hh.gene.max_fold_changes.gS 6 -> 10 only',optimization_records=str(O),iterations_by_case=iterations,total_additional_iterations=sum(iterations.values()),response_comparison=report);(target/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
manifest={}
for p in stage.rglob('*'):
 if p.is_file() and p.relative_to(stage)!=Path('data/provenance.json'):
  q=target/p.relative_to(stage);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==sha;manifest[str(q.relative_to(target))]=sha
images=list(valid['images'])+['nd_comparison.png']
for name in images:
 with Image.open(target/name) as im:im.verify()
assert (target/'nd_comparison.pdf').is_file();assert float(np.load(target/'data/costs.npy')[0])==s['cost'];assert int(np.load(target/'data/best_adam.npz')['iter'])==s['adam_iter']
result=dict(folder=str(target),cost=s['cost'],new_mapping_initial_cost=new['total'],old_mapping_cost=old['total'],improvement_percent_from_training_start=100*(1-s['cost']/new['total']),improvement_percent_from_old_model=100*(1-s['cost']/old['total']),old_ND=old['groups'],new_mapping_ND=new['groups'],trained_ND=s['metrics']['groups'],images=images,total_additional_iterations=sum(iterations.values()),parameters_and_adam_verified=True,authorized_config_change_only=True,files=manifest)
(O/'export_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2),flush=True)
