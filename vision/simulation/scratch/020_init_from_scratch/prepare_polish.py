from pathlib import Path
import json
O=Path(__file__).resolve().parent
assert json.loads((O/'mbar_initial_01_0.001/progress.json').read_text())['complete']
rows=[]
for name in ['mbar_cases.json','refine_cases.json','extend_cases.json']:
 for case in json.loads((O/name).read_text()):
  p=O/case['label']/'progress.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d.get('complete'):rows.append(d)
winner=min(rows,key=lambda d:d['best_cost']);source=O/winner['case']['label']/'run'
p=O/'extend_cases.json';cases=json.loads(p.read_text());assert len(cases)==3
common=dict(source=str(source),native=False,resume_adam=True,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.})
cases.extend([dict(common,label='polish_00_adam',lr=.001,n_iter=64,backtrack=False),dict(common,label='polish_01_backtrack',lr=.0003,n_iter=32,backtrack=True)])
tmp=O/'extend_cases.pending';tmp.write_text(json.dumps(cases,indent=2)+'\n');tmp.replace(p)
(O/'polish_seed.json').write_text(json.dumps(dict(source=str(source),cost=winner['best_cost'],additional_iterations=[64,32]),indent=2)+'\n')
print(json.dumps(dict(source=str(source),cost=winner['best_cost'],added_cases=2)))
