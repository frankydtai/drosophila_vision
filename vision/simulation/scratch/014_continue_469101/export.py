from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[3]
selection=json.loads((O/'selected.json').read_text());valid=json.loads((O/'render_result.json').read_text());stage=Path(selection['run'])
assert valid['selected_cost']==selection['cost']<469100.78125
assert valid['parameter_roundtrip_exact'] and len(valid['images'])==7
provenance=json.loads((stage/'data/provenance.json').read_text())
target=ROOT/'vision/simulation/0_runs/point_hh'/f'{provenance["job"]}-continued_optimization-loss-{selection["cost"]:.0f}'
assert not target.exists(),target
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
shutil.copytree(stage,target)
provenance.update(optimization_origin_run=str(ROOT/'vision/simulation/0_runs/point_hh/34435454-adam_reset-n_iter-20-lr-0.003-final_cost-469101'),optimization_records=str(O),baseline_cost=469100.78125,improvement_percent=100*(1-selection['cost']/469100.78125))
(target/'data/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
manifest={}
for p in stage.rglob('*'):
 if p.is_file() and p.relative_to(stage)!=Path('data/provenance.json'):
  q=target/p.relative_to(stage);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==sha
  manifest[str(q.relative_to(target))]=sha
for name in valid['images']:
 with Image.open(target/name) as im:im.verify()
assert float(np.load(target/'data/costs.npy')[0])==selection['cost']
assert int(np.load(target/'data/best_adam.npz')['iter'])==selection['adam_iter']
result=dict(folder=str(target),cost=selection['cost'],improvement_percent=provenance['improvement_percent'],images=list(valid['images']),parameters_and_adam_verified=True,protected_files_unchanged=True,files=manifest)
(O/'export_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
