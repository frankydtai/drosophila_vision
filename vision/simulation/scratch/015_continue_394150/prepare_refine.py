from pathlib import Path
import json
O=Path(__file__).resolve().parent
screen=json.loads((O/'screen_cases.json').read_text());rows=[]
for case in screen:
 p=O/case['label']/'progress.json';d=json.loads(p.read_text());assert d.get('complete'),case['label'];assert (p.parent/'run/data/best_param.npz').is_file();rows.append(d)
winner=min(rows,key=lambda d:d['best_cost']);source=O/winner['case']['label']/'run'
cases=[dict(label=f'refine_{i:02d}_{"resume" if resume else "reset"}_backtrack',source=str(source),resume_adam=resume,lr=.0003,n_iter=16,backtrack=True) for i,resume in enumerate([True,False])]
(O/'refine_cases.json').write_text(json.dumps(cases,indent=2)+'\n');(O/'screen_winner.json').write_text(json.dumps(dict(label=winner['case']['label'],cost=winner['best_cost'],run=str(source)),indent=2)+'\n');print(json.dumps(winner['case']),winner['best_cost'])
