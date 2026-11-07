from pathlib import Path
import json,hashlib,subprocess
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;cases=json.loads((O/'cases.json').read_text());results={};rows=[]
for c in cases:
 d=json.loads((O/c['label']/'metrics.json').read_text());results[c['label']]=d;r=dict(case=c['label'],kind=c['kind'],fold=c['upper_fold'],lower_per_contact_nS=d['nominal_per_contact_gS_lower_nS'],upper_per_contact_nS=d['nominal_per_contact_gS_upper_nS'],loss=d['total'])
 for k,g in d['groups'].items():
  cell=k.rsplit('_',1)[-1]
  for name,v in g.items():r[cell+'_'+name]=v
 rows.append(r)
pd.DataFrame(rows).to_csv(O/'all_comparisons.csv',index=False)
base=results['upper_only_6'];proposal=results['configured_10'];audit_base=results['upper_only_6__release_audit'];assert base['total']==audit_base['total'];assert base['groups']==audit_base['groups']
# Native changed configuration and algebraically equivalent runtime map agree.
runtime=results['symmetric_10'];assert abs(runtime['total']-proposal['total'])/runtime['total']<1e-4
assert pd.read_csv(O/'upper_only_6/membrane_conductances.csv').equals(pd.read_csv(O/'configured_10/membrane_conductances.csv'))
assert base['parameter_summary']['active_gS_edges']==proposal['parameter_summary']['active_gS_edges']==27824
changes={}
for k,g in base['groups'].items():
 a=proposal['groups'][k];changes[k]=dict(nd_peak_change_percent=100*(a['nd_peak_mV']/g['nd_peak_mV']-1),pd_peak_change_percent=100*(a['pd_peak_mV']/g['pd_peak_mV']-1),before=g,after=a)
summary=dict(source=base['case']['run'],no_training=True,proposed_config_change='point_hh.gene.max_fold_changes.gS: 6.0 -> 10.0',original_loss=base['total'],proposed_loss=proposal['total'],loss_improvement_percent=100*(1-proposal['total']/base['total']),original_per_contact_range_nS=[.05,1.8],proposed_per_contact_range_nS=[.03,3.],response_changes=changes,baseline_Mi4_release=audit_base['mi4_output'],temporary_runtime_vs_native_loss_difference=proposal['total']-runtime['total'])
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
import sys
sys.path[:0]=[str(O/'copy/vision/simulation'),str(O/'copy/vision')]
import import_bootstrap
from task.mbar.sti_spec import COST_ALIGNED_FIRST_STI_MS
b=np.load(O/'upper_only_6/curves.npz');a=np.load(O/'configured_10/curves.npz');fig,axes=plt.subplots(2,2,figsize=(11,6.8),sharex=True)
for row,cell in enumerate(['T4a','T4b']):
 for col,direction in enumerate(['PD','ND']):
  key=next(k[:-7] for k in b.files if k.endswith(f'_{cell}_{direction}__model'));gt=b[key+'__gt'];assert np.array_equal(gt,a[key+'__gt']);t=(np.arange(len(gt))-COST_ALIGNED_FIRST_STI_MS)/1000
  ax=axes[row,col];ax.plot(t,gt,color='0.4',label='Target',linewidth=2);ax.plot(t,b[key+'__model'],label='Current fold 6',color='#2774ae',linewidth=1.8);ax.plot(t,a[key+'__model'],label='Proposed fold 10',color='#d47727',linewidth=1.8);ax.set_title(f'{cell} {direction}');ax.set_ylabel('Response above baseline [mV]');ax.grid(alpha=.2)
  if row==1:ax.set_xlabel('Time from first stimulus [s]')
axes[0,0].legend();fig.suptitle('Same trained weights; no retraining; only gS fold range changed');fig.tight_layout();fig.savefig(O/'proposed_fold10_PD_ND.png',dpi=180);fig.savefig(O/'proposed_fold10_PD_ND.pdf');plt.close(fig)
labels=['upper_only_6','symmetric_8','configured_10','symmetric_12','symmetric_16'];folds=[results[k]['case']['upper_fold'] for k in labels];fig,axes=plt.subplots(2,2,figsize=(10,6.6))
axes[0,0].plot(folds,[results[k]['total'] for k in labels],marker='o');axes[0,0].set_ylabel('Original loss (a_gt2)')
for cell,color in [('T4a','#2774ae'),('T4b','#d47727')]:
 key='mbar_DSI_bright_'+cell
 for ax,field in [(axes[0,1],'nd_peak_mV'),(axes[1,0],'pd_peak_mV'),(axes[1,1],'nd_pd_ratio')]:ax.plot(folds,[results[k]['groups'][key][field] for k in labels],marker='o',color=color,label=cell)
axes[0,1].set_ylabel('ND peak [mV]');axes[1,0].set_ylabel('PD peak [mV]');axes[1,1].set_ylabel('ND / PD peak ratio')
for ax in axes.flat:ax.set_xlabel('gS max_fold_changes');ax.grid(alpha=.2)
axes[0,1].legend();fig.suptitle('Existing symmetric fold formula; same weights, no retraining');fig.tight_layout();fig.savefig(O/'fold_range_tradeoff.png',dpi=180);fig.savefig(O/'fold_range_tradeoff.pdf');plt.close(fig)
protected=json.loads((O/'protected_hashes.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());before=set((O/'git_before.txt').read_text().splitlines());after=set(subprocess.check_output(['git','status','--short'],text=True).splitlines());assert before==after,(sorted(after-before),sorted(before-after))
from PIL import Image
for name in ['proposed_fold10_PD_ND.png','fold_range_tradeoff.png']:
 with Image.open(O/name) as im:im.verify()
audit=dict(no_training=True,evaluations=len(cases),formal_files_unchanged=len(protected),git_status_unchanged=True,baseline_reproduced_exactly=True,proposal_uses_unmodified_model_implementation=True,native_vs_runtime_relative_loss_difference=abs(proposal['total']-runtime['total'])/runtime['total'],connectivity_unchanged=True,membrane_conductances_unchanged=True,proposed_config_difference='one scalar: gS max_fold_changes 6 -> 10');(O/'final_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(audit=audit,summary=summary),indent=2))
