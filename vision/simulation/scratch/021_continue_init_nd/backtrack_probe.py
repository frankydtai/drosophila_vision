from pathlib import Path
import json,numpy as np
from response_metrics import O,setup,evaluate
setup();source=json.loads((O/'campaign_state.json').read_text())['source'];metrics,curves=evaluate(source,O/'backtrack_probe/best_state.npz')
(O/'backtrack_probe/metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez(O/'backtrack_probe/curves.npz',**{key+'__'+label:v for key,pair in curves.items() for label,v in zip(['model','gt'],pair)})
print(json.dumps(metrics,indent=2),flush=True)
