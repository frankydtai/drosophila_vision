"""Compare static sums with moving bars, including the legacy degree/mid bug."""
from __future__ import annotations
import os
from pathlib import Path
import sys
from dataclasses import replace
from types import SimpleNamespace
import json

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM.parent))
os.chdir(SIM)
import import_bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train
from figure.plot import load_best
from figure.mbar import _traces_from_forward
from task.mbar.pack import bar_specs_from_task
from task.mbar.gt import motion_preference
from neuron.forward import inject_a_sti_mid, forward_v
from build_hex import DEG

SOURCE = SIM / "0_runs/hp_lp/30223811-run_init_from-n_iter-3000-tasks-sbar-a_lsd-1"
MOVING = SIM / "0_runs/hp_lp/34213914-run_init_from-n_iter-0-tasks-mbar-part_cost_scales-mbar-0,mbar_DSI-1"
OUT = SIM / "scratch/t4a_sbar_mbar_diagnosis"


@torch.no_grad()
def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    session, z, _ = load_best(str(MOVING), verbose=True)
    pack = session.packs["mbar"]["bright"]
    params = train.params_from_z(z, session)
    sigma = float(params["a_sti_mid"].reshape(-1)[0])
    print('sigma', sigma, 'side',session.connectome.meta.get('side'), flush=True)
    all_rows, metrics, inputs = [], [], []
    dark = session.packs['mbar']['dark']
    for name, contrast, p in (
        ("Legacy moving Gaussian", 'bright', replace(pack, bar_axis_distance=pack.bar_axis_distance * float(DEG))),
        ("Moving Gaussian, corrected units", 'bright', pack),
        ("Moving rectangle, no Gaussian", 'bright', replace(pack, bar_axis_distance=None)),
        ("Dark NC: legacy moving Gaussian", 'dark', replace(dark, bar_axis_distance=dark.bar_axis_distance * float(DEG))),
        ("Dark NC: corrected units", 'dark', dark),
    ):
        print('running',name,flush=True)
        specs = bar_specs_from_task(session, "mbar", contrast)
        tokens = [s.token for s in specs]
        params = train.params_from_z(z, session)
        v = train.forward_pack(session, params, p.i_sti, p)
        windows, cells, side, n_hex, t_onset, single = _traces_from_forward(
            session, "mbar", contrast, v.cpu().numpy(), specs, tokens)
        for cell in ("T4a", "T4b"):
            for b, (spec, token) in enumerate(zip(specs,tokens)):
                key=cell,token
                values=windows.ca_mean_cell[key]
                idx=(windows.cell_idxs==cells.index(cell)) & (windows.t0_bn[b]>=0)
                nodes=np.flatnonzero(idx)
                baseline=float(v[b, t_onset-1, nodes].mean())
                times=(np.arange(len(values))-windows.before_t[token])*float(session.delta_ms)
                pref=motion_preference(side,cell,spec.direction,contrast).pd_nd
                response=values-baseline
                all_rows.append(pd.DataFrame(dict(case=name,cell=cell,direction=pref,time_ms=times,vm_mv=response)))
                mask=(pack.entry_bs==b)&torch.tensor(
                    [session.connectome.cells[int(x)]==cell for x in session.connectome.node_cells[pack.entry_nodes]],
                    device=pack.entry_bs.device)
                entries=torch.nonzero(mask).flatten()
                peaks=[]
                for e in entries:
                    n=int(pack.entry_nodes[e]);start=int(pack.dsi_t0s[e]);stop=start+int(pack.dsi_n_ts[e])
                    peaks.append(float((v[b,start:stop,n]-v[b,t_onset-1,n]).max()))
                metrics.append(dict(case=name,cell=cell,direction=pref,
                                    peak_mean_trace_mv=float(response.max()),
                                    mean_node_peak_in_dsi_window_mv=float(np.mean(peaks)),
                                    baseline_mv=baseline,n_nodes=len(nodes)))
                if cell=='T4a':
                    inputs.append((name,spec.direction,np.asarray(v[b,:,nodes]).mean(axis=1)))
        # Effective input at an R1-6 node closest to x=y=0.
        effective=inject_a_sti_mid(p.i_sti,train.params_from_z(z,session),p)
        conn=session.connectome
        r_nodes=[int(n) for n in conn.sti_nodes if conn.cells[int(conn.node_cells[n])]=='R1-6']
        n=min(r_nodes,key=lambda n:float(conn.us[n])**2+float(conn.vs[n])**2)
        for b,spec in enumerate(specs):
            tt=(np.arange(effective.shape[1])-pack.t_onset)*float(session.delta_ms)
            pd.DataFrame(dict(time_ms=tt,current_pa=effective[b,:,n].cpu().numpy())).to_csv(
                OUT / f"input_{name.replace(' ','_').replace(',','')}_{spec.direction}.csv",index=False)
        del effective,v
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)

    # Same static pulses: sum outputs separately, versus drive their sum once.
    static, zs, _ = load_best(str(SOURCE), verbose=True)
    sp=static.packs['sbar']['bright']
    ps=train.params_from_z(zs,static)
    effective=inject_a_sti_mid(sp.i_sti,ps,sp)
    from train.cost import entries_by_part
    part_idxs,part_keys=entries_by_part(sp)
    mids=(-2,-1,0,1,2)
    entries={m:torch.nonzero(part_idxs==part_keys.index(f'sbar_bright_T4a_mid{m:+d}')).flatten() for m in mids}
    nodes={m:sp.entry_nodes[entries[m]] for m in mids}
    for m in mids:assert torch.equal(nodes[m],nodes[0])
    common_nodes=nodes[0]
    print('static same readout nodes',len(common_nodes),flush=True)
    for duration in (160,40):
        pulse_drive=effective.clone()
        if duration !=160:
            pulse_drive[:,sp.t_onset+int(duration/static.delta_ms):]=pulse_drive[:,0:1]
        # Simulate independent sources and a combined 5-position drive.
        plain=SimpleNamespace(t_onset=sp.t_onset)
        lone=forward_v(static,train.params_from_z(zs,static),pulse_drive,pack=plain)
        n_t=effective.shape[1]+int(320/static.delta_ms)
        joint=torch.zeros((2,n_t,effective.shape[2]),dtype=effective.dtype,device=effective.device)
        joint[:]=effective[0,0]
        sums=np.zeros((2,n_t))
        for d,order in enumerate((mids,tuple(reversed(mids)))):
            for m in order:
                es=entries[m];b=int(sp.entry_bs[es[0]])
                assert bool((sp.entry_bs[es]==b).all())
                delay=int(order.index(m)*80/static.delta_ms)
                # Source drive is finite pulse; baseline outside is unchanged.
                delta=pulse_drive[b]-pulse_drive[b,0]
                joint[d,delay:delay+delta.shape[0]]+=delta
                source=lone[b,:,common_nodes].mean(dim=1).cpu().numpy()
                source=source-source[sp.t_onset-50:sp.t_onset].mean()
                sums[d,delay:delay+len(source)]+=source
        combined=forward_v(static,train.params_from_z(zs,static),joint,pack=plain)
        for d,pref in enumerate(('PD','ND')):
            model=combined[d,:,common_nodes].mean(dim=1).cpu().numpy()
            base=float(model[sp.t_onset-50:sp.t_onset].mean())
            t=(np.arange(n_t)-sp.t_onset)*float(static.delta_ms)
            for label,values in ((f'{duration}-ms flashes: sum outputs',sums[d]),
                                 (f'{duration}-ms flashes: joint network',model-base)):
                all_rows.append(pd.DataFrame(dict(case=label,cell='T4a',direction=pref,time_ms=t,vm_mv=values)))
                metrics.append(dict(case=label,cell='T4a',direction=pref,
                                    peak_mean_trace_mv=float(values[t>=0].max())))
        del combined,lone,pulse_drive,joint
    data=pd.concat(all_rows,ignore_index=True)
    data.to_csv(OUT/'response_traces.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'peak_comparison.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    (OUT/'settings.json').write_text(json.dumps(dict(sigma=sigma,degree_per_mid=float(DEG),speed_deg_s=56.25,
        legacy_sigma_deg=sigma,corrected_sigma_deg=sigma*float(DEG),
        legacy_fwhm_ms=2.35482*sigma/56.25*1000,
        corrected_fwhm_ms=2.35482*sigma*float(DEG)/56.25*1000),indent=2))
    names=['160-ms flashes: sum outputs','160-ms flashes: joint network',
           'Moving Gaussian, corrected units','40-ms flashes: sum outputs',
           '40-ms flashes: joint network','Legacy moving Gaussian']
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,name in zip(axes.flat,names):
        for pref,color in [('PD','#d1493f'),('ND','#2768ad')]:
            q=data[(data.case==name)&(data.cell=='T4a')&(data.direction==pref)]
            ax.plot(q.time_ms,q.vm_mv,color=color,label=pref,lw=2)
        ax.set_title(name);ax.set_xlim(-100,600);ax.set_ylim(-12,44);ax.axhline(0,color='0.7',lw=.7);ax.grid(alpha=.15);ax.legend(frameon=False)
        ax.set_xlabel('Time (ms; local onset for moving)');ax.set_ylabel('Baseline-subtracted T4a (mV)')
    fig.suptitle('Same inherited parameters: stimulus and superposition controls',fontsize=15)
    fig.savefig(OUT/'comparison.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
