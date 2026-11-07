from pathlib import Path
import json
O=Path(__file__).resolve().parent
for p in sorted(O.glob('*/progress.json')):
 d=json.loads(p.read_text());print(json.dumps(dict(case=p.parent.name,n=len(d['records']),best_total=d['best_cost'],best_nd=d['best_nd'],best_parts=d['best_parts'],unconstrained_best=d['unconstrained_best_cost'],complete=d.get('complete',False),stop=d.get('stop_reason'))))
p=O/'source_metrics.json'
if p.exists():print('BASELINE_RESPONSE',json.dumps(json.loads(p.read_text())['groups']))
