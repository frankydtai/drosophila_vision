"""Compare independently produced native A and B results; no simulation here."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

OUT = Path(__file__).resolve().parent
def load(path):
    with np.load(path,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}
def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def period(y):
    p,_=find_peaks(y,prominence=max(float(np.ptp(y))*.25,.1),distance=5)
    return None if len(p)<3 else float(np.diff(p).mean())

report={}
for run_id in ('34400581','34401154'):
    sub=OUT/run_id
    p=load(sub/'matched_inputs.npz')
    ai,bi=load(sub/'A_initial.npz'),load(sub/'B_initial.npz')
    a,b=load(sub/'A_observation.npz'),load(sub/'B_observation.npz')
    am,bm=[json.loads((sub/f'{k}_manifest.json').read_text()) for k in ('A','B')]
    assert am['source_unchanged'] and bm['source_unchanged']
    assert bm['no_A_state_read'] and bm['no_Point_HH_import']
    assert bm['matched_inputs_sha256']==hashlib.sha256((sub/'matched_inputs.npz').read_bytes()).hexdigest()
    # Native B returns one extra physical update. Compare the actual common
    # t=0..1200 ms without changing or relabeling either native execution.
    n=min(a['voltage_mV'].shape[0],b['voltage_mV'].shape[0])
    assert np.array_equal(a['time_ms'][:n],b['time_ms'][:n])
    va,vb=a['voltage_mV'][:n],b['voltage_mV'][:n]
    tt=a['time_ms'][:n]
    tail=(tt>=200)&(tt<=1200)
    cells=p['cells'].tolist();nc=p['node_cell']
    state_keys={'v':'v','m':'HH_m','h':'HH_h','n':'HH_n'}
    rows=[]
    for i in range(len(nc)):
        row=dict(node=i,cell=cells[nc[i]])
        for ak,bk in state_keys.items():
            row[f'A_{ak}']=float(ai[ak][i]);row[f'B_{ak}']=float(bi[bk][i])
            row[f'B_minus_A_{ak}']=float(bi[bk][i]-ai[ak][i])
        row.update(A_tail_std_mV=float(va[tail,i].std()),B_tail_std_mV=float(vb[tail,i].std()),
                   A_tail_ptp_mV=float(np.ptp(va[tail,i])),B_tail_ptp_mV=float(np.ptp(vb[tail,i])))
        rows.append(row)
    write_csv(sub/'all_neuron_comparison.csv',rows)
    erows=[dict(edge=i,source=int(p['source'][i]),target=int(p['target'][i]),
        A_s=float(ai['s'][i]),B_s=float(bi['Syn_s'][i]),B_minus_A_s=float(bi['Syn_s'][i]-ai['s'][i]))
        for i in range(len(p['source']))]
    write_csv(sub/'all_edge_initial_release.csv',erows)
    populations=[]
    for ci,cell in enumerate(cells):
        ix=np.flatnonzero(nc==ci)
        ya=va[:,ix].mean(1);yb=vb[:,ix].mean(1)
        populations.append(dict(cell=cell,n_neurons=len(ix),
            A_population_std_mV=float(ya[tail].std()),B_population_std_mV=float(yb[tail].std()),
            A_median_single_neuron_ptp_mV=float(np.median(np.ptp(va[tail][:,ix],axis=0))),
            B_median_single_neuron_ptp_mV=float(np.median(np.ptp(vb[tail][:,ix],axis=0)))))
    write_csv(sub/'all_celltype_comparison.csv',populations)
    reps=[]
    fig,axs=plt.subplots(3,2,figsize=(13,9),sharex=True)
    fig2,axs2=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for ri,cell in enumerate(('L1','Mi1','Tm3')):
        ix=np.flatnonzero(nc==cells.index(cell))
        # Selection uses A only and is fixed for the B comparison.
        node=int(ix[np.argmax(np.ptp(va[tail][:,ix],axis=0))])
        rep=dict(node=node,cell=cell,A_v0=float(ai['v'][node]),B_v0=float(bi['v'][node]),
            A_ptp_mV=float(np.ptp(va[tail,node])),B_ptp_mV=float(np.ptp(vb[tail,node])),
            A_period_ms=period(va[tail,node]),B_period_ms=period(vb[tail,node]))
        reps.append(rep)
        for axes,ys,kind in ((axs,(va[:,node],vb[:,node]),f'same neuron #{node}'),
                             (axs2,(va[:,ix].mean(1),vb[:,ix].mean(1)),'population mean')):
            ymin=min(float(y.min()) for y in ys);ymax=max(float(y.max()) for y in ys)
            pad=max((ymax-ymin)*.08,.05)
            for col,(name,y,color) in enumerate(zip(('A: project original','B: Ripple original'),ys,('#2874b6','#d24b36'))):
                ax=axes[ri,col]
                ax.plot(tt,y,color=color,lw=.85)
                ax.set_ylim(ymin-pad,ymax+pad)
                ax.set_xlim(0,1200)
                ax.set_title(f'{cell} {kind} | {name}',fontsize=10)
                ax.set_ylabel('Raw voltage (mV)');ax.grid(alpha=.2)
                if ri==2:ax.set_xlabel('Time after each initialization endpoint (ms)')
    status=bm['status']
    title=(f'{run_id} | independent native initializations | dt = 1 ms\n'
           f'A: original pre_steady + {am["physical_pre_ms"]} ms pre-period; '
           f'B: Ripple cold start + segmented integration ({bm["physical_pre_ms"]} ms total; {status})\n'
           'Same graph, physical parameters and constant background; each row has identical A/B axis limits')
    for fig,name in ((fig,'single_neuron_native_A_B.png'),(fig2,'population_native_A_B.png')):
        fig.suptitle(title,fontsize=11)
        fig.tight_layout(rect=(0,0,1,.90));fig.savefig(sub/name,dpi=160);plt.close(fig)
    # A short common window makes waveform width and sampling visible.
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,r in zip(axes,reps):
        node=r['node'];mask=tt<=250
        ax.plot(tt[mask],va[mask,node],label='A: project original',lw=1.2,color='#2874b6')
        ax.plot(tt[mask],vb[mask,node],label='B: Ripple original',lw=1.2,color='#d24b36')
        ax.set_title(f'{r["cell"]} same neuron #{node} | '
            f'V0 A={r["A_v0"]:.3f}, B={r["B_v0"]:.3f} mV',fontsize=10)
        ax.set_ylabel('Raw V (mV)');ax.grid(alpha=.2)
    axes[0].legend(loc='best',frameon=False);axes[-1].set_xlabel('Time after each initialization endpoint (ms)')
    fig.suptitle(f'{run_id}: independent native A/B | dt=1 ms | B status: {status}',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(sub/'single_neuron_native_overlay.png',dpi=160);plt.close(fig)
    report[run_id]=dict(A=am,B=bm,representatives=reps,celltypes=len(cells),nodes=len(nc),
        initial_voltage_max_abs_difference_mV=float(np.max(abs(bi['v']-ai['v']))),
        initial_voltage_rms_difference_mV=float(np.sqrt(np.mean((bi['v']-ai['v'])**2))),
        all_states_finite=bool(all(np.isfinite(v).all() for v in (*ai.values(),*bi.values()))),
        comparison_window_ms=[0,int(tt[-1])],
        limitation='Native A float32 and native B float64 and their numerical kernels are retained; '
                   'this compares independent native pipelines, not initialization alone under an identical solver.')
(OUT/'comparison_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:dict(A_ms=v['A']['physical_pre_ms'],B_ms=v['B']['physical_pre_ms'],
    B_status=v['B']['status'],reps=v['representatives']) for k,v in report.items()},indent=2),flush=True)
