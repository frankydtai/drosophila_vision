"""Compare two 300-iteration branches from spread100 on identical test entries."""
from pathlib import Path
import csv,json,runpy
import numpy as np
SIM=Path(__file__).resolve().parents[1]
OUT=SIM/'scratch/compare_spread100_ab_vs_abcd_20261105'
OUT.mkdir(exist_ok=True)
BASE=SIM/'0_runs/hp_lp'
AB=next(BASE.glob('34381383-*'))
AB_EVAL=BASE/'34382310-t4abcd-from-34381383-iter-0'
ABCD=BASE/'34382547-t4abcd-direct-from-34376877-iter-300'
INITIAL=BASE/'34382547-t4abcd-direct-from-34376877-iter-0'
ns=runpy.run_path(str(SIM/'scratch/t4abcd_from_34381383_20261105/analyze.py'))
ns['torch'].set_num_threads(1)


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    from matplotlib import pyplot as plt
    models={}
    for label,run in [('spread100',INITIAL),('AB300',AB),('ABCD300',ABCD)]:
        with np.load(run/'data/best_param.npz',allow_pickle=True) as f:
            models[label]={k:f[k].copy() for k in f.files}
    with np.load(AB_EVAL/'data/best_param.npz',allow_pickle=True) as f:
        assert all(np.array_equal(f[k],models['AB300'][k]) for k in f.files)
    param_rows=[]
    for key,arr in models['AB300'].items():
        if key in ('cells','pairs'):continue
        names=(models['AB300']['pairs'] if key.startswith('syn_strength') else
               models['AB300']['cells'] if arr.size==len(models['AB300']['cells']) else ['shared'])
        for i,name in enumerate(names):
            a=float(arr.ravel()[i]);b=float(models['ABCD300'][key].ravel()[i]);v=float(models['spread100'][key].ravel()[i])
            param_rows.append(dict(parameter=key,cell=str(name),spread100=v,AB300=a,ABCD300=b,
                                   AB_minus_spread=a-v,ABCD_minus_spread=b-v,ABCD_minus_AB=b-a))
    write_csv('parameter_comparison.csv',param_rows)
    info,curves,metrics={},{},[]
    for label,run in [('AB300',AB_EVAL),('ABCD300',ABCD)]:
        rows,curve,result=ns['evaluate']('300',run)
        for row in rows:row['model']=label
        metrics.extend(rows);curves[label]=curve;info[label]=result
        with np.load((AB if label=='AB300' else ABCD)/'data/best_adam.npz',allow_pickle=True) as f:
            info[label]['best_training_iteration']=int(f['iter'])
    write_csv('fit_metrics.csv',metrics)
    summary=[]
    for cell in ns['CELLS']:
        row={'cell':cell}
        for label in ['AB300','ABCD300']:
            rr=[r for r in metrics if r['model']==label and r['cell']==cell and r['scope']=='cost_entries']
            row[label+'_rmse_mv']=float(np.sqrt(np.mean([r['rmse_mv']**2 for r in rr])))
            for r in rr:row[label+'_'+r['pd_nd']+'_peak_mv']=r['peak_delta_mv']
            pd=row[label+'_PD_peak_mv'];nd=row[label+'_ND_peak_mv']
            row[label+'_peak_DSI']=(pd-nd)/(pd+nd) if pd+nd else None
            row[label+'_raw_PD_minus_ND_sum']=info[label]['raw_pd_minus_nd_sum'][f'mbar_DSI_bright_{cell}']
        summary.append(row)
    write_csv('cell_comparison.csv',summary)
    for scope in ['cost_entries','center_hex']:
        fig,axes=plt.subplots(4,2,figsize=(10,10),sharex=True,sharey='row')
        for i,cell in enumerate(ns['CELLS']):
            for j,direction in enumerate(['PD','ND']):
                ax=axes[i,j]
                for label,color in [('AB300','#3569b0'),('ABCD300','#da7426')]:
                    t,pred,gt=curves[label][cell,direction,scope]
                    if label=='AB300':ax.plot(t,gt,color='black',lw=1.8,label='GT')
                    ax.plot(t,pred,color=color,lw=1.4,label=label+' (best)')
                ax.axvline(0,color='gray',lw=.8,ls=':');ax.axhline(0,color='gray',lw=.5)
                ax.set_title(cell+' '+direction);ax.set_xlim(-150,576);ax.grid(alpha=.15)
                if j==0:ax.set_ylabel('Vm - cell onset bias (mV)')
                if i==3:ax.set_xlabel('Time from local stimulus onset (ms)')
        axes[0,0].legend(frameon=False,fontsize=8)
        fig.suptitle('Same spread100 source: AB-only 300 vs joint ABCD 300\nIdentical four-direction evaluation, '+scope.replace('_',' '),fontsize=12)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/f'waveforms_{scope}.png',dpi=150);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(dict(source='34376877 spread100',AB_training_run=str(AB),ABCD_training_run=str(ABCD),AB_parameter_identity_verified=True,evaluation='Same four directions, bright, cost_mid=0 cross union; no parameter updates.',models=info,cells=summary),indent=2)+'\n')
    print(json.dumps(dict(models=info,cells=summary),indent=2),flush=True)

if __name__=='__main__':main()
