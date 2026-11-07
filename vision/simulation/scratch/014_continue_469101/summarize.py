from pathlib import Path
import json
O=Path(__file__).resolve().parent;rows=[]
for p in sorted(O.glob('*/progress.json')):
 d=json.loads(p.read_text());r=d['records'];rows.append(dict(label=d['case']['label'],n=len(r),initial=d.get('initial_cost'),best=d.get('best_cost'),last=r[-1]['cost'] if r else None,terminal=d.get('terminal_cost'),complete=d.get('complete',False),lr=d.get('current_lr'),stop=d.get('stop_reason')))
(O/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(json.dumps(r))
