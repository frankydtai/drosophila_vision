from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from response_metrics import O
from task.mbar.sti_spec import COST_ALIGNED_FIRST_STI_MS
selection=json.loads((O/'selected.json').read_text());run=Path(selection['run'])
source=np.load(O/'source_curves.npz');after=np.load(run.parent/'response_curves.npz')
base=json.loads((O/'source_metrics.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,6.6),sharex=True)
for row,cell in enumerate(['T4a','T4b']):
 for col,direction in enumerate(['PD','ND']):
  key=next(k[:-7] for k in source.files if k.endswith(f'_{cell}_{direction}__model'))
  before=source[key+'__model'];current=after[key+'__model'];gt=source[key+'__gt'];assert np.array_equal(gt,after[key+'__gt'])
  t=(np.arange(len(gt))-COST_ALIGNED_FIRST_STI_MS)/1000
  ax=axes[row,col];ax.plot(t,gt,color='0.4',label='Target',linewidth=2);ax.plot(t,before,color='#df8b27',label='Before',linewidth=1.6);ax.plot(t,current,color='#1675b8',label='After',linewidth=1.8)
  ax.axhline(0,color='0.7',linewidth=.6);ax.set_title(f'{cell} {direction}');ax.grid(alpha=.2);ax.set_ylabel('Response above baseline [mV]')
  if row==1:ax.set_xlabel('Time from first stimulus [s]')
axes[0,0].legend();fig.suptitle(f'Init mode: ND continuation, original loss {base["total"]:.2f} → {selection["cost"]:.2f}')
fig.tight_layout();fig.savefig(run/'nd_comparison.png',dpi=180);fig.savefig(run/'nd_comparison.pdf');plt.close(fig)
print('ND comparison saved',flush=True)
