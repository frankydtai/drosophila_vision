from pathlib import Path
import json
O=Path(__file__).resolve().parent
rows=[]
for case in json.loads((O/'refine_cases.json').read_text()):
 d=json.loads((O/case['label']/'progress.json').read_text());assert d.get('complete'),case['label'];rows.append(d)
winner=min(rows,key=lambda d:d['best_cost'])
source=O/winner['case']['label']/'run'
cases=[dict(label='extend_00_resume_backtrack',source=str(source),resume_adam=True,lr=.0003,initial_lr=max(1e-7,min(.0003,winner['current_lr'])),n_iter=32,backtrack=True)]
(O/'extend_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
(O/'refine_winner.json').write_text(json.dumps(dict(label=winner['case']['label'],cost=winner['best_cost'],run=str(source)),indent=2)+'\n')
print(json.dumps(dict(source=str(source),cost=winner['best_cost'],additional_steps=32)))
