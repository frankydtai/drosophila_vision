from pathlib import Path
import json,hashlib,subprocess
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;cases=json.loads((O/'cases.json').read_text());results={};rows=[]
for case in cases:
 d=json.loads((O/case['label']/'metrics.json').read_text());results[case['label']]=d;row=dict(case=case['label'],loss=d['total'])
 for key,g in d['groups'].items():
  cell=key.rsplit('_',1)[-1]
  for name,value in g.items():row[cell+'_'+name]=value
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'all_comparisons.csv',index=False)
rule=results['rule_best'];init=results['init_best'];masked=results['init_weights_close_unresolved']
summary=dict(rule=rule,init=init,unresolved_mask_only=masked,mask_only_loss_improvement_percent=100*(1-masked['total']/init['total']),controlled_cases=len(cases),all_evaluations_no_training=True)
(O/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
r=np.load(O/'rule_best/curves.npz');i=np.load(O/'init_best/curves.npz');fig,axes=plt.subplots(2,2,figsize=(11,6.8),sharex=True)
# The loss pack aligns each response window's first stimulus to this constant.
import sys
sys.path[:0]=[str(O/'copy/vision/simulation'),str(O/'copy/vision')]
import import_bootstrap
from task.mbar.sti_spec import COST_ALIGNED_FIRST_STI_MS
for row,cell in enumerate(['T4a','T4b']):
 for col,direction in enumerate(['PD','ND']):
  key=next(k[:-7] for k in r.files if k.endswith(f'_{cell}_{direction}__model'));gt=r[key+'__gt'];assert np.array_equal(gt,i[key+'__gt']);t=(np.arange(len(gt))-COST_ALIGNED_FIRST_STI_MS)/1000
  ax=axes[row,col];ax.plot(t,gt,color='0.35',linewidth=2,label='Target');ax.plot(t,r[key+'__model'],color='#2874a6',linewidth=1.8,label='Rule trained');ax.plot(t,i[key+'__model'],color='#d87328',linewidth=1.8,label='Init trained');ax.set_title(f'{cell} {direction}');ax.set_ylabel('Response above baseline [mV]');ax.grid(alpha=.2)
  if row==1:ax.set_xlabel('Time from first stimulus [s]')
axes[0,0].legend();fig.suptitle(f'Trained Rule vs Init; original loss {rule["total"]:.2f} vs {init["total"]:.2f}');fig.tight_layout();fig.savefig(O/'rule_vs_init_PD_ND.png',dpi=180);fig.savefig(O/'rule_vs_init_PD_ND.pdf');plt.close(fig)
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
before=set((O/'git_before.txt').read_text().splitlines());after=set(subprocess.check_output(['git','status','--short'],text=True).splitlines());assert before==after,(sorted(after-before),sorted(before-after))
audit=dict(no_training=True,formal_files_unchanged=len(protected),git_status_unchanged=True,stored_costs_reproduced_exactly=True,cases=len(cases),record_folder=str(O));(O/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(audit=audit,mask_only_loss_improvement_percent=summary['mask_only_loss_improvement_percent'],original_losses=dict(rule=rule['total'],init=init['total']),masked_loss=masked['total']),indent=2))
