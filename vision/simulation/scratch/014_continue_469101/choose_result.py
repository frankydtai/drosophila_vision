from pathlib import Path
import json,numpy as np
O=Path(__file__).resolve().parent;rows=[]
for p in sorted(O.glob('*/progress.json')):
 d=json.loads(p.read_text());assert d.get('complete'),p
 run=p.parent/'run';saved=float(np.load(run/'data/costs.npy')[0]);assert saved==d['best_cost']
 rows.append(dict(label=d['case']['label'],run=str(run),cost=saved,adam_iter=d['selected_adam_iter'],case=d['case']))
winner=min(rows,key=lambda d:d['cost']);assert winner['cost']<469100.78125
(O/'selected.json').write_text(json.dumps(winner,indent=2)+'\n');(O/'all_results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(winner,indent=2))
