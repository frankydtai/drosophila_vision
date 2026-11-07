from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image
O=Path(__file__).resolve().parent;ROOT=O.parents[3]
validation=json.loads((O/'render_result.json').read_text());replay=json.loads((O/'replay_result.json').read_text())
assert validation['selected_cost']==replay['selected_cost']==469100.78125
assert validation['parameter_roundtrip_exact'] and replay['all_20_costs_and_gradient_norms_match_diagnostic_exactly']
assert len(validation['images'])==7
stage=O/'run'
target=ROOT/'vision/simulation/0_runs/point_hh'/f'{replay["replay_job"]}-adam_reset-n_iter-20-lr-0.003-final_cost-469101'
assert not target.exists(),target
protected=json.loads((O/'protected_hashes.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
shutil.copytree(stage,target)
shutil.copy2(O/f'replay-{replay["replay_job"]}.log',target/'replay.log')
shutil.copy2(O/'render_result.json',target/'data/figure_validation.json')
manifest={}
for p in stage.rglob('*'):
 if p.is_file():
  q=target/p.relative_to(stage);sha=hashlib.sha256(p.read_bytes()).hexdigest()
  assert q.is_file() and hashlib.sha256(q.read_bytes()).hexdigest()==sha,q
  manifest[str(q.relative_to(target))]=sha
for name in validation['images']:
 with Image.open(target/name) as im:im.verify()
assert float(np.load(target/'data/costs.npy')[0])==469100.78125
assert int(np.load(target/'data/best_adam.npz')['iter'])==20
result=dict(folder=str(target),selected_cost=469100.78125,images=list(validation['images']),stage_files_exactly_copied=True,protected_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items()),files=manifest)
(O/'export_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
