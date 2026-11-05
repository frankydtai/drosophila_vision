"""Reevaluate unchanged checkpoints after fixing vertical anatomy; no training."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
SIM=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SIM))
import import_bootstrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import figure.plot as plot
from train.param import params_from_z
from train.cost import forward_pack,gt_affine_from_pack,_pack_cost_parts_from_v_readout,_weighted_mean_cost_parts
from neuron.readout import pack_traces
from task.mbar.pack import dsi_values
from task.mbar.sti_spec import build_mbar_signals
from task.mbar.pack import bar_specs_from_task,mbar_cost_sti_hexes
from task.mbar.sti_geo import view_bounds,cost_motion_view_deg
from task.sbar.sti_geo import sti_hexes

OUT=Path(__file__).resolve().parent
RUNS={
 'AB300':SIM/'0_runs/hp_lp/34382310-t4abcd-from-34381383-iter-0',
 'ABCD300_wrong_supervision':SIM/'0_runs/hp_lp/34382547-t4abcd-direct-from-34376877-iter-300',
}
CELLS=['T4a','T4b','T4c','T4d']


def evaluate(run):
    path=run/'data/best_param.npz';before=hashlib.sha256(path.read_bytes()).hexdigest()
    session,z,saved_cost=plot.load_best(str(run))
    pack=session.packs['mbar']['bright'];params=params_from_z(z,session)
    with torch.no_grad():
        trace=forward_pack(session,params,pack.i_sti,pack)
        pred=pack_traces(trace,pack)
        a,b=gt_affine_from_pack(params,pack,session)
        target=a[:,None]*pack.gts+b[:,None]
        parts=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,session,a,b,pred,trace=trace),session)
        loss=sum(float(v) for v in parts.values())
        dsis={k:float(v) for k,v in dsi_values(pack,trace).items()}
    pred=pred.cpu().numpy();target=target.cpu().numpy();bias=b.cpu().numpy()
    weights=pack.cost_scales.cpu().numpy()
    t=np.arange(pred.shape[1])*session.delta_ms-324.
    groups={};metrics=[]
    for key in dict.fromkeys(pack.entry_part_keys):
        mask=np.array([k==key for k in pack.entry_part_keys])&(weights>0)
        cell,direction=key.split('_')[-2:]
        p=np.average(pred[mask],axis=0,weights=weights[mask]);g=np.average(target[mask],axis=0,weights=weights[mask])
        base=float(np.average(bias[mask],weights=weights[mask]));delta=p-np.mean(p[t<0])
        metrics.append(dict(cell=cell,pd_nd=direction,rmse_mv=float(np.sqrt(np.mean((p-g)**2))),
                            peak_delta_mv=float(delta[t>=0].max()),n_entries=int(mask.sum())))
        groups[cell,direction]=(t,p-base,g-base)
    assert before==hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(legacy_saved_cost=saved_cost,corrected_eval_cost=loss,raw_PD_minus_ND_sum=dsis,
                corrected_geometry=session.train_opts['mbar_sti_opts']['geometry_version'],
                parameter_file_unchanged_sha256=before,groups=metrics),groups,session,pack


def check_legacy_stimulus_permutation(session,pack):
    # Read the pre-fix cached top-hat currents without allowing them into new cache.
    old_cache=Path(session.train_opts['network_json']).parent/'mbar_cache/1addb78bb648ced7.npz'
    with np.load(old_cache) as f:old=f['i_sti_hex']
    specs=bar_specs_from_task(session,'mbar','bright')
    hexes=mbar_cost_sti_hexes(session.connectome,specs,cost_mid=0)
    view=cost_motion_view_deg(view_bounds(sti_hexes(session.connectome)),hexes,[s.direction for s in specs])
    sti=build_mbar_signals(session.connectome,specs=specs,n_t=pack.i_sti.shape[1],t_onset=pack.t_onset,
                          delta_ms=session.delta_ms,bar_dist=4,multi_bar=False,i_baseline=20.,i_sti=40.,
                          use_cache=True,use_a_sti_mid=True,view_deg=view)
    expected=old[[0,1,3,2]]
    assert np.array_equal(sti.i_sti_hex,expected)
    return dict(horizontal_exactly_unchanged=True,corrected_up_equals_old_down=True,
                corrected_down_equals_old_up=True,legacy_cache=str(old_cache))


def main():
    torch.set_num_threads(1)
    report={};curves={}
    for name,run in RUNS.items():
        print('Evaluating',name,flush=True)
        info,groups,session,pack=evaluate(run)
        report[name]=info;curves[name]=groups
        if name=='AB300':report['stimulus_check']=check_legacy_stimulus_permutation(session,pack)
    legacy=json.loads((SIM/'scratch/t4abcd_from_34381383_20261105/summary.json').read_text())['0']
    comparison=[]
    for cell in CELLS:
        info=report['AB300'];rows=[r for r in info['groups'] if r['cell']==cell]
        peaks={r['pd_nd']:r['peak_delta_mv'] for r in rows}
        comparison.append(dict(cell=cell,legacy_rmse_mv=legacy['cells'][cell]['rmse_mv'],
                               corrected_rmse_mv=float(np.sqrt(np.mean([r['rmse_mv']**2 for r in rows]))),
                               legacy_PD_minus_ND_sum=legacy['raw_pd_minus_nd_sum'][f'mbar_DSI_bright_{cell}'],
                               corrected_PD_minus_ND_sum=info['raw_PD_minus_ND_sum'][f'mbar_DSI_bright_{cell}'],
                               corrected_PD_peak_mv=peaks['PD'],corrected_ND_peak_mv=peaks['ND']))
    report['unchanged_AB_checkpoint_before_after']=comparison
    with (OUT/'unchanged_AB_before_after.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    fig,axes=plt.subplots(4,2,figsize=(10,10),sharex=True,sharey='row')
    for i,cell in enumerate(CELLS):
        for j,d in enumerate(['PD','ND']):
            ax=axes[i,j];t,p,g=curves['AB300'][cell,d]
            old=curves['AB300'][cell,('ND' if d=='PD' else 'PD')][1] if cell in ['T4c','T4d'] else p
            ax.plot(t,g,color='black',label='GT',lw=1.7)
            ax.plot(t,old,color='#999999',ls='--',label='Old direction labels',lw=1.3)
            ax.plot(t,p,color='#2b8c5a',label='Corrected directions (same params)',lw=1.4)
            ax.axvline(0,color='gray',lw=.7,ls=':');ax.axhline(0,color='gray',lw=.5)
            ax.set_title(cell+' '+d);ax.set_xlim(-150,576);ax.grid(alpha=.15)
            if j==0:ax.set_ylabel('Vm - cell onset bias (mV)')
            if i==3:ax.set_xlabel('Time from local stimulus onset (ms)')
    axes[0,0].legend(frameon=False,fontsize=7)
    fig.suptitle('34381383 checkpoint: direction fix only, zero parameter updates',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'unchanged_AB_corrected_direction.png',dpi=150)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
