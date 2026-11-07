from pathlib import Path
import json
O=Path(__file__).resolve().parent;rows=[]
for case in json.loads((O/'mbar_cases.json').read_text()):
 d=json.loads((O/case['label']/'progress.json').read_text());rows.append(d)
winner=min(rows,key=lambda d:d['best_cost']);assert winner.get('complete'),winner['case']['label'];source=O/winner['case']['label']/'run'
cases=[dict(label=f'refine_{i:02d}_{"resume" if resume else "reset"}',source=str(source),native=False,resume_adam=resume,lr=.0003,n_iter=32,backtrack=True,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.}) for i,resume in enumerate([True,False])]
(O/'refine_seed.json').write_text(json.dumps(dict(source=str(source),cost=winner['best_cost'],other_comparisons_pending=[d['case']['label'] for d in rows if not d.get('complete')]),indent=2)+'\n')
(O/'refine_cases.json').write_text(json.dumps(cases,indent=2)+'\n');print(json.dumps(dict(source=str(source),cost=winner['best_cost'])))
