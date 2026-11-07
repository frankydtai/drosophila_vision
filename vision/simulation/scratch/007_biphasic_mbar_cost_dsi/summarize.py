"""Archive tables, scientific plots, source audit, and research conclusions."""
from pathlib import Path
import json,csv,hashlib,ast,os
import numpy as np
OUT=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
refs=json.loads((OUT/'references.json').read_text());rows=[]
for fn in ['intervention_results.json','followup_results.json','validation_results.json','k001_extension_results.json']:
 for r in json.loads((OUT/fn).read_text()):r['origin']=fn;rows.append(r)
by={r['case']['name']:r for r in rows};base=by['saved_baseline'];BC=base['cost'];KEYS=['mbar_DSI_bright_T4a','mbar_DSI_bright_T4b']
with (OUT/'all_results.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['case','k_reference','cost','cost_change_percent','DSI_raw_T4a','DSI_raw_T4b','peak_DSI_T4a','peak_DSI_T4b','DSI_baseline_subtracted_T4a','DSI_baseline_subtracted_T4b','T4a_PD_peak_mV','T4a_ND_peak_mV','T4b_PD_peak_mV','T4b_ND_peak_mV','mean_only_cost','sd_band_cost','parameters_json'])
 for r in rows:
  c=r['case'];label=c.get('k_label','HH_saved');baseline=refs[label]['cost'];w.writerow([c['name'],label,r['cost'],100*(r['cost']/baseline-1),*[r['dsi_raw'][k] for k in KEYS],*[r['peak_metrics'][c]['peak_DSI'] for c in ['T4a','T4b']],*[r['dsi_baseline_subtracted'][k] for k in KEYS],*[r['peak_metrics'][c][kind] for c in ['T4a','T4b'] for kind in ['PD_peak','ND_peak']],r['mean_only_cost'],r['sd_band_cost'],json.dumps(c)])
# Keep two DSIs separate: raw objective diagnostics and normalized peak metric.
dose_rows=[(0,base)]+[(r['case']['dose'],r) for r in rows if r['case']['name'].startswith('fine_negative_') or r['case']['name'].startswith('excitatory_negative_')];dose_rows.sort(key=lambda a:a[0]);xx=np.array([a for a,r in dose_rows]);fig,ax=plt.subplots(1,3,figsize=(14,4))
ax[0].plot(xx,[r['cost']/BC for a,r in dose_rows],'o-',color='tab:purple');ax[0].axhline(1,color='k',lw=.8,ls='--');ax[0].set_ylabel('mbar cost / native HH cost');ax[0].set_title('Large negative lobes overcorrect')
for c,key,col in zip(['T4a','T4b'],KEYS,['tab:blue','tab:orange']):
 ax[1].plot(xx,[r['dsi_raw'][key] for a,r in dose_rows],'o-',label=c,color=col);ax[2].plot(xx,[r['peak_metrics'][c]['peak_DSI'] for a,r in dose_rows],'o-',label=c,color=col)
ax[1].axhline(0,color='k',lw=.8);ax[1].set_ylabel('Raw PD-minus-ND voltage sum');ax[1].set_title('Integrated direction contrast improves');ax[2].set_ylabel('(PD peak - ND peak) / (PD peak + ND peak)');ax[2].set_title('Peak separation changes much less');ax[1].legend();ax[2].legend()
for a in ax:a.set_xlabel('Injected HPLP negative-lobe amplitude');a.grid(alpha=.2)
fig.suptitle('1 ms HH, Mi1/Tm3 signals delivered to T4 only; no retraining',fontsize=12);fig.tight_layout();fig.savefig(OUT/'amplitude_tradeoff.png',dpi=180);plt.close(fig)
# Pick a bounded-grid example with both raw DSIs positive and improved cost if present.
good=[r for r in rows if r['case'].get('k_label','HH_saved')=='HH_saved' and r['cost']<BC and all(r['dsi_raw'][k]>0 for k in KEYS)];improved=[r for r in rows if r['case'].get('k_label','HH_saved')=='HH_saved' and r['cost']<BC and all(r['dsi_raw'][k]>base['dsi_raw'][k] for k in KEYS) and all(r['peak_metrics'][c]['peak_DSI']>base['peak_metrics'][c]['peak_DSI'] for c in ['T4a','T4b'])];best=min(good,key=lambda r:r['cost']) if good else min(improved,key=lambda r:r['cost']);selected=best['case']['name']
fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
curves=[('saved_baseline','Native HH','k'),(selected,selected,'tab:green'),('excitatory_negative_1','Full negative lobe','tab:red'),('HPLP_matched','HPLP matched input','tab:blue')]
for i,c in enumerate(['T4a','T4b']):
 for j,d in enumerate(['right_bright_w1','left_bright_w1']):
  a=ax[i,j]
  for name,label,color in curves:
   f=OUT/'HPLP_matched_curves.npz' if name=='HPLP_matched' else OUT/'curves'/(name+'.npz');z=np.load(f);a.plot(z['time__'+d],z[c+'__'+d],label=label,color=color,lw=1.5)
  a.axhline(0,color='.5',lw=.6);a.set_xlim(-150,650);a.set_title(c+' '+('PD' if (i==0 and j==0) or (i==1 and j==1) else 'ND'));a.grid(alpha=.15)
  if j==0:a.set_ylabel('Baseline-relative voltage (mV)')
  if i==1:a.set_xlabel('Time from local first stimulus (ms)')
ax[0,0].legend(fontsize=8);fig.suptitle('Signal restoration improves tails; excessive negative lobes overshoot');fig.tight_layout();fig.savefig(OUT/'t4_waveforms.png',dpi=180);plt.close(fig)
# Audit: original physics and data unchanged; another task only added a peak DSI helper.
m=json.loads((OUT/'manifest.json').read_text());audit={}
for f,h in m['source_hashes'].items():audit[f]=dict(initial=h,current=hashlib.sha256(Path(f).read_bytes()).hexdigest(),unchanged=hashlib.sha256(Path(f).read_bytes()).hexdigest()==h)
def functions(data):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(data).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
old=functions((OUT/'source_snapshot/mbar_pack_initial.py').read_text());new=functions(Path(OUT.parents[1]/'3_task/4_mbar/4_pack.py').read_text());audit['mbar_pack_existing_functions_unchanged']=all(new.get(k)==v for k,v in old.items());audit['mbar_pack_added_functions']=list(new.keys()-old.keys());audit['saved_parameter_files_unchanged']={}
for label,run in [('HH',Path(m['HH_run'])),('HPLP',Path(m['HPLP_run']))]:
 for name in ['train_opts.json','best_param.npz']:audit['saved_parameter_files_unchanged'][label+'/'+name]=hashlib.sha256((run/'data'/name).read_bytes()).digest()==hashlib.sha256((OUT/'source_snapshot'/label/name).read_bytes()).digest()
assert audit['mbar_pack_existing_functions_unchanged'] and all(audit['saved_parameter_files_unchanged'].values());(OUT/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
selection=dict(total_cases=len(rows),selected_case=selected,selected_cost=best['cost'],selected_dsi_raw=best['dsi_raw'],selected_peak_metrics=best['peak_metrics'],simultaneous_positive_DSI_and_lower_cost_cases=[r['case']['name'] for r in good]);(OUT/'summary_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print(json.dumps(selection,indent=2))
# The same lobe change has opposite cost effects at different fixed k values.
fig,ax=plt.subplots(2,3,figsize=(14,7))
for i,(label,bname,prefix) in enumerate([('saved k=0.04453','saved_baseline','saved'),('fixed k=0.01','k001_baseline','k001')]):
 br=by[bname]
 rr=dose_rows if prefix=='saved' else [(0,br)]+[(r['case']['dose'],r) for r in rows if r['case']['name'].startswith('k001_negative_') or r['case']['name']=='k001_small_negative']
 rr.sort(key=lambda z:z[0]);xx=[x for x,r in rr];ax[i,0].plot(xx,[r['cost']/br['cost'] for x,r in rr],'o-',color='tab:purple');ax[i,0].axhline(1,color='k',ls='--',lw=.7);ax[i,0].set_ylabel(label+'\ncost / own baseline')
 for c,k,col in zip(['T4a','T4b'],KEYS,['tab:blue','tab:orange']):
  ax[i,1].plot(xx,[r['dsi_raw'][k] for x,r in rr],'o-',label=c,color=col);ax[i,2].plot(xx,[r['peak_metrics'][c]['peak_DSI'] for x,r in rr],'o-',label=c,color=col)
 ax[i,1].axhline(0,color='k',lw=.7)
 for a in ax[i]:a.set_xlabel('HPLP negative-lobe amplitude');a.grid(alpha=.2)
ax[0,0].set_title('Cost: lower is better');ax[0,1].set_title('Raw voltage-sum DSI');ax[0,2].set_title('Normalized peak DSI');ax[0,1].legend();fig.suptitle('Negative lobes and release kinetics must be tuned together (1 ms; no training)');fig.tight_layout();fig.savefig(OUT/'kinetics_amplitude_comparison.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
for i,c in enumerate(['T4a','T4b']):
 for j,d in enumerate(['right_bright_w1','left_bright_w1']):
  for name,label,col in [('k001_baseline','k=0.01 baseline','k'),('k001_negative_1','k=0.01 + full negative lobe','tab:green'),('saved_baseline','Saved HH (k=0.04453)','tab:gray'),('HPLP_matched','HPLP matched input','tab:blue')]:
   f=OUT/'HPLP_matched_curves.npz' if name=='HPLP_matched' else OUT/'curves'/(name+'.npz');z=np.load(f);ax[i,j].plot(z['time__'+d],z[c+'__'+d],label=label,color=col,lw=1.5)
  ax[i,j].set_title(c+' '+('PD' if i==j else 'ND'));ax[i,j].set_xlim(-150,650);ax[i,j].axhline(0,color='.5',lw=.6);ax[i,j].grid(alpha=.15)
  if j==0:ax[i,j].set_ylabel('Baseline-relative voltage (mV)')
  if i==1:ax[i,j].set_xlabel('Time from local first stimulus (ms)')
ax[0,0].legend(fontsize=8);fig.suptitle('At k=0.01, adding a negative lobe reduces prolonged responses');fig.tight_layout();fig.savefig(OUT/'k001_t4_waveforms.png',dpi=180);plt.close(fig)
