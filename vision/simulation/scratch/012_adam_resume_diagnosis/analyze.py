from pathlib import Path
import numpy as np,json,hashlib,os
O=Path(__file__).resolve().parent
src=O/'inputs/34431163/data'; dst=O/'inputs/34432972/data'
a=np.load(src/'best_costs.npy');b=np.load(dst/'best_costs.npy');best=int(np.argmin(a));replayed=a[best:best+len(b)]
report={'source_best_history_index':best,'source_best_adam_iter':int(np.load(src/'best_adam.npz')['iter']),'new_best_adam_iter':int(np.load(dst/'best_adam.npz')['iter']),'options_identical':json.loads((src/'train_opts.json').read_text())==json.loads((dst/'train_opts.json').read_text()),'resume_initial':float(b[0]),'new_best_cost':float(np.load(dst/'costs.npy')[0]),'new_best_history_index':int(np.argmin(b)),'new_last_pre_update_cost':float(b[-1]),'early_replay_relative_error_max_first_10':float(np.max(np.abs(b[:10]-replayed[:10])/replayed[:10])),'early_replay_relative_error_max_first_20':float(np.max(np.abs(b[:20]-replayed[:20])/replayed[:20])),'cases':{}}
for mode in ['resume','reset','resume_small','reset_small']:
 p=O/f'{mode}.json'
 if p.exists():report['cases'][mode]=json.loads(p.read_text())
manifest=json.loads((O/'source_hashes.json').read_text())
report['production_changed_files']=[p for p,h in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
(O/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
os.environ['MPLCONFIGDIR']=str(O/'matplotlib_analysis')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4))
ax=axes[0];ax.plot(np.arange(len(b)),b/1e6,label='34432972: resumed');ax.plot(np.arange(len(replayed)),replayed/1e6,'--',label='34431163: after its best point');ax.scatter(np.argmin(b),b.min()/1e6,color='black',s=20,label='Saved best');ax.set_xlabel('Updates from source best point');ax.set_ylabel('Loss (million)');ax.set_title('Resuming initially replays the previous path');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axes[1]
for mode,d in report['cases'].items():
 vals=[r['cost'] for r in d['records']]
 if 'final_cost' in d:vals.append(d['final_cost'])
 ax.plot(np.arange(len(vals)),np.array(vals)/1e6,label=f'{mode}, lr={d["lr"]}')
ax.axhline(b[0]/1e6,color='grey',linestyle=':',label='Initial loss');ax.set_xlabel('Updates');ax.set_ylabel('Loss (million)');ax.set_title('Temporary controls (CPU)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(O/'adam_diagnosis.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
for mode,d in report['cases'].items():print(mode,len(d['records']),d.get('final_cost'),d.get('best_cost'),d.get('probes'))
