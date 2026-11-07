from pathlib import Path
import json,numpy as np
from response_metrics import O,setup,evaluate
setup();source=json.loads((O/'campaign_state.json').read_text())['source'];out={}
for label in ['screen_0003','screen_0001']:
 path=O/'early_response_probe'/(label+'.npz');metrics,_=evaluate(source,snapshot=path,gs_fold=10.);metrics.pop('parameter_tables')
 with np.load(path) as a:assert metrics['total']==float(a['cost'])
 out[label]=metrics
(O/'early_response_probe/metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
