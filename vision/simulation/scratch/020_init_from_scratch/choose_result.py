from pathlib import Path
import json,numpy as np
O=Path(__file__).resolve().parent;rows=[]
for name in ['mbar_cases.json','refine_cases.json','extend_cases.json']:
 p=O/name
 if not p.exists():continue
 for case in json.loads(p.read_text()):
  assert Path(case['source']).is_relative_to(O),case['source']
  folder=O/case['label'];d=json.loads((folder/'progress.json').read_text());assert d.get('complete'),folder
  run=folder/'run';opts=json.loads((run/'data/train_opts.json').read_text())
  assert opts['tasks']==['mbar'] and opts['point_hh']['g_context_mode']=='init'
  saved=float(np.load(run/'data/costs.npy')[0]);assert saved==d['best_cost']
  rows.append(dict(label=case['label'],run=str(run),cost=saved,adam_iter=d['selected_adam_iter'],case=case))
assert rows
baseline=json.loads((O/json.loads((O/'mbar_cases.json').read_text())[0]['label']/'progress.json').read_text())['initial_cost']
winner=min(rows,key=lambda d:d['cost']);assert winner['cost']<baseline
(O/'selected.json').write_text(json.dumps(winner,indent=2)+'\n');(O/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n')
(O/'mbar_baseline.json').write_text(json.dumps(dict(initial_cost=baseline),indent=2)+'\n');print(json.dumps(winner,indent=2))
