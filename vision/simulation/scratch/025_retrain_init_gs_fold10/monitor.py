from pathlib import Path
import json
O=Path(__file__).resolve().parent
for p in sorted(O.glob('*/progress.json')):
 d=json.loads(p.read_text());r=d['records'][-1] if d['records'] else {};print(json.dumps(dict(case=p.parent.name,n=len(d['records']),planned=d['case']['n_iter'],best=d['best_cost'],nd=d['best_nd'],last=r.get('cost'),lr=d['current_lr'],complete=d.get('complete',False),stop=d['stop_reason'])))
