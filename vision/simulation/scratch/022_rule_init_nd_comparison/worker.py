from pathlib import Path
import json,sys,numpy as np
from response_metrics import O,setup,evaluate
case=json.loads((O/'cases.json').read_text())[int(sys.argv[1])];folder=O/case['label'];folder.mkdir(exist_ok=True)
setup();metrics,curves=evaluate(case['run'],override=case['override']);metrics['case']=case
expected=float(np.load(Path(case['run'])/'data/costs.npy')[0]);metrics['stored_cost']=expected
if case['override'] is None:assert metrics['total']==expected,(metrics['total'],expected)
ions,syn,summary=metrics.pop('parameter_tables');metrics['parameter_summary']=summary
import pandas as pd
pd.DataFrame(ions).to_csv(folder/'membrane_conductances.csv',index=False);syn.to_csv(folder/'synapse_conductances.csv',index=False)
np.savez(folder/'curves.npz',**{key+'__'+label:v for key,pair in curves.items() for label,v in zip(['model','gt'],pair)})
(folder/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2),flush=True)
