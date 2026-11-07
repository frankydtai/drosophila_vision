from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from response_metrics import O
from task.mbar.sti_spec import COST_ALIGNED_FIRST_STI_MS
s=json.loads((O/'selected.json').read_text());run=Path(s['run']);old=np.load(O/'old_mapping_curves.npz');start=np.load(O/'new_mapping_curves.npz');after=np.load(run.parent/'response_curves.npz');fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for row,cell in enumerate(['T4a','T4b']):
 for col,direction in enumerate(['PD','ND']):
  key=next(k[:-7] for k in old.files if k.endswith(f'_{cell}_{direction}__model'));gt=old[key+'__gt'];assert np.array_equal(gt,after[key+'__gt']) and np.array_equal(gt,start[key+'__gt']);t=(np.arange(len(gt))-COST_ALIGNED_FIRST_STI_MS)/1000
  ax=axes[row,col];ax.plot(t,gt,color='0.4',label='Target',linewidth=2);ax.plot(t,old[key+'__model'],color='#2774ae',label='Old fold 6',linewidth=1.5);ax.plot(t,start[key+'__model'],color='#d47727',label='Fold 10 before retraining',linewidth=1.5);ax.plot(t,after[key+'__model'],color='#268c51',label='Fold 10 retrained',linewidth=1.8);ax.set_title(f'{cell} {direction}');ax.set_ylabel('Response above baseline [mV]');ax.grid(alpha=.2)
  if row==1:ax.set_xlabel('Time from first stimulus [s]')
axes[0,0].legend(fontsize=8);fig.suptitle(f'Init gS fold 6 → 10, retrained; final loss {s["cost"]:.2f}');fig.tight_layout();fig.savefig(run/'nd_comparison.png',dpi=180);fig.savefig(run/'nd_comparison.pdf');plt.close(fig)
print('ND and PD before/after comparison saved.',flush=True)
