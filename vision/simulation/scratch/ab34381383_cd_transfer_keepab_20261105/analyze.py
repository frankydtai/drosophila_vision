"""AB checkpoint transfer: matched ABCD evaluation and native AB retention."""
from pathlib import Path
import sys,json,csv,runpy
import numpy as np
SIM=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SIM))
import import_bootstrap
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figure.plot as plot
from train.implementation import load_best_node_vals,session_z_from_node_vals
from train.param import params_from_z
from train.cost import forward_pack,gt_affine_from_pack,_pack_cost_parts_from_v_readout,_weighted_mean_cost_parts
from neuron.readout import pack_traces
from task.mbar.pack import dsi_values
OUT=Path(__file__).resolve().parent
runpy.run_path(str(OUT/'check_code.py'))
SOURCE=next((SIM/'0_runs/hp_lp').glob('34381383-*/data/best_param.npz')).parent.parent
CELLS=['T4a','T4b','T4c','T4d']


def write_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(session,z):
    pack=session.packs['mbar']['bright'];params=params_from_z(z,session)
    with torch.no_grad():
        trace=forward_pack(session,params,pack.i_sti,pack)
        pred=pack_traces(trace,pack)
        a,b=gt_affine_from_pack(params,pack,session)
        gt=a[:,None]*pack.gts+b[:,None]
        parts=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,session,a,b,pred,trace=trace),session)
        dsis={k:float(v) for k,v in dsi_values(pack,trace).items()}
    p=pred.cpu().numpy();g=gt.cpu().numpy();bias=b.cpu().numpy()
    weights=pack.cost_scales.cpu().numpy();t=np.arange(p.shape[1])*session.delta_ms-324.
    rows=[];curves={}
    for key in dict.fromkeys(pack.entry_part_keys):
        mask=np.array([k==key for k in pack.entry_part_keys])&(weights>0)
        cell,d=key.split('_')[-2:]
        pm=np.average(p[mask],axis=0,weights=weights[mask]);gm=np.average(g[mask],axis=0,weights=weights[mask])
        bm=float(np.average(bias[mask],weights=weights[mask]))
        pd=pm-pm[t<0].mean();gd=gm-gm[t<0].mean()
        rows.append(dict(cell=cell,pd_nd=d,n_entries=int(mask.sum()),
                         rmse_mv=float(np.sqrt(np.mean((pm-gm)**2))),
                         response_rmse_mv=float(np.sqrt(np.mean((pm[t>=0]-gm[t>=0])**2))),
                         peak_delta_mv=float(pd[t>=0].max()),gt_peak_delta_mv=float(gd[t>=0].max())))
        curves[cell,d]=(t,pm-bm,gm-bm)
    cells={}
    for cell in dict.fromkeys(r['cell'] for r in rows):
        rr=[r for r in rows if r['cell']==cell]
        peaks={r['pd_nd']:r['peak_delta_mv'] for r in rr}
        cells[cell]=dict(rmse_mv=float(np.sqrt(np.mean([r['rmse_mv']**2 for r in rr]))),
                         peaks_mv=peaks,gt_peaks_mv={r['pd_nd']:r['gt_peak_delta_mv'] for r in rr},
                         peak_DSI=(peaks['PD']-peaks['ND'])/(peaks['PD']+peaks['ND']),
                         raw_PD_minus_ND_sum=dsis[f'mbar_DSI_bright_{cell}'])
    return dict(cost=sum(float(v) for v in parts.values()),cells=cells,groups=rows),curves


def evaluate(run):
    session,z,saved=plot.load_best(str(run));info,curves=stats(session,z)
    assert np.isclose(info['cost'],saved,rtol=1e-4,atol=.1),(info['cost'],saved)
    info['saved_cost']=saved
    with np.load(run/'data/best_adam.npz',allow_pickle=True) as f:info['best_iteration']=int(f['iter'])
    # Evaluate on the exact original horizontal AB stimulus/entry selection too.
    native_session,_,_=plot.load_best(str(SOURCE))
    values,cells,pairs=load_best_node_vals(str(run))
    native_session,nz=session_z_from_node_vals(values,cells,pairs,native_session)
    info['native_AB'],_=stats(native_session,nz)
    return info,curves


def params_compare(baseline,trained):
    models={}
    for label,run in [('AB_initial',baseline),('after300',trained)]:
        with np.load(run/'data/best_param.npz',allow_pickle=True) as f:models[label]={k:f[k].copy() for k in f.files}
    rows=[]
    for key,a in models['AB_initial'].items():
        if key in ['cells','pairs']:continue
        b=models['after300'][key]
        labels=models['AB_initial']['pairs'] if key.startswith('syn_strength') else models['AB_initial']['cells'] if a.size==len(models['AB_initial']['cells']) else ['shared']
        for name,x,y in zip(labels,a.ravel(),b.ravel()):
            rows.append(dict(parameter=key,cell=str(name),AB_initial=float(x),after300=float(y),delta=float(y-x),percent_change=float((y-x)/abs(x)*100) if abs(x)>1e-8 else None))
    write_csv('parameter_changes.csv',rows)
    return rows


def main():
    torch.set_num_threads(1)
    run=Path(sys.argv[1]).resolve();iterations=int(sys.argv[2])
    current,curves=evaluate(run)
    record=dict(run=str(run),iterations=iterations,source=str(SOURCE),geometry='fafb_y_ventral_v2',evaluation=current)
    (OUT/f'result_{iterations}.json').write_text(json.dumps(record,indent=2)+'\n')
    write_csv(f'metrics_{iterations}.csv',current['groups'])
    if iterations==0:
        with np.load(SOURCE/'data/best_param.npz',allow_pickle=True) as a,np.load(run/'data/best_param.npz',allow_pickle=True) as b:
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
        fig,axes=plt.subplots(2,2,figsize=(9,5),sharex=True,sharey='row')
        for i,c in enumerate(['T4c','T4d']):
            for j,d in enumerate(['PD','ND']):
                t,p,g=curves[c,d];ax=axes[i,j]
                ax.plot(t,g,color='black',label='GT');ax.plot(t,p,color='#3569b0',label='AB checkpoint, 0 updates')
                ax.set_title(c+' '+d);ax.set_xlim(-150,576);ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'CD_baseline.png',dpi=150);plt.close(fig)
        (OUT/'baseline_run.txt').write_text(str(run)+'\n')
    else:
        baseline=Path(sys.argv[3]).resolve();initial,initial_curves=evaluate(baseline)
        rows=[]
        for c in CELLS:
            a,b=initial['cells'][c],current['cells'][c]
            rows.append(dict(cell=c,rmse_before_mv=a['rmse_mv'],rmse_after_mv=b['rmse_mv'],
                             PD_peak_before_mv=a['peaks_mv']['PD'],ND_peak_before_mv=a['peaks_mv']['ND'],
                             PD_peak_after_mv=b['peaks_mv']['PD'],ND_peak_after_mv=b['peaks_mv']['ND'],
                             peak_DSI_before=a['peak_DSI'],peak_DSI_after=b['peak_DSI']))
        write_csv('before_after.csv',rows);params_compare(baseline,run)
        fig,axes=plt.subplots(4,2,figsize=(10,10),sharex=True,sharey='row')
        for i,c in enumerate(CELLS):
            for j,d in enumerate(['PD','ND']):
                t,p,g=curves[c,d];_,old,_=initial_curves[c,d];ax=axes[i,j]
                ax.plot(t,g,color='black',lw=1.7,label='GT')
                ax.plot(t,old,color='#3569b0',lw=1.4,label='AB checkpoint (0 updates)')
                ax.plot(t,p,color='#da7426',lw=1.4,label=f'ABCD continuation (best @{current["best_iteration"]})')
                ax.set_title(c+' '+d);ax.set_xlim(-150,576);ax.axvline(0,color='gray',ls=':',lw=.7);ax.grid(alpha=.15)
                if j==0:ax.set_ylabel('Vm - cell onset bias (mV)')
                if i==3:ax.set_xlabel('Time from local stimulus onset (ms)')
        axes[0,0].legend(fontsize=7,frameon=False);fig.tight_layout();fig.savefig(OUT/'before_after.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4));costs=np.load(run/'data/best_costs.npy');ax.plot(costs,color='black',label='Total')
        with np.load(run/'data/best_costs_by_part.npz') as f:
            for c in CELLS:
                ks=[k for k in f.files if f'_{c}_' in k and not k.startswith('mbar_DSI')]
                ax.plot(np.sum([f[k] for k in ks],axis=0),label=c)
        ax.set(xlabel='Iteration',ylabel='Loss contribution',yscale='log');ax.legend();fig.tight_layout();fig.savefig(OUT/'loss_by_cell.png',dpi=150);plt.close(fig)
        (OUT/'comparison.json').write_text(json.dumps(dict(before=initial,after=current,cells=rows),indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
