from pathlib import Path
import json
O=Path(__file__).resolve().parent;rows=[]
for name in ['mbar_cases.json','refine_cases.json']:
 for case in json.loads((O/name).read_text()):
  d=json.loads((O/case['label']/'progress.json').read_text());rows.append(d)
winner=min(rows,key=lambda d:d['best_cost']);assert winner.get('complete'),winner['case']['label']
source=O/winner['case']['label']/'run'
cases=[dict(label='extend_00_adam',source=str(source),native=False,resume_adam=True,lr=.003,n_iter=80,backtrack=False,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.}),dict(label='extend_01_backtrack',source=str(source),native=False,resume_adam=True,lr=.001,n_iter=32,backtrack=True,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.})]
cases.append(dict(label='extend_02_reset',source=str(source),native=False,resume_adam=False,lr=.001,n_iter=40,backtrack=False,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.}))
assert not (O/'extend_cases.json').exists()
(O/'extend_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(O/'extend_seed.json').write_text(json.dumps(dict(source=str(source),cost=winner['best_cost'],pending_comparisons=[d['case']['label'] for d in rows if not d.get('complete')]),indent=2)+'\n')
print(json.dumps(dict(source=str(source),cost=winner['best_cost'],additional_cases=3)))
