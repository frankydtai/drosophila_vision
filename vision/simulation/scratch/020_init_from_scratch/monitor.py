from pathlib import Path
import json
O=Path(__file__).resolve().parent
for label in ['polish_00_adam','polish_01_backtrack']:
 p=O/label/'progress.json'
 if p.exists():
  d=json.loads(p.read_text());print(json.dumps(dict(label=label,n=len(d['records']),planned=d['case']['n_iter'],best=d['best_cost'],last=d['records'][-1]['cost'] if d['records'] else None,complete=d.get('complete',False),stop=d.get('stop_reason'))))
p=O/'export_result.json'
if p.exists():print(p.read_text())
