"""Read saved models; all derived artifacts remain beside this script."""
from pathlib import Path
import sys, os, json, csv, hashlib, gc
OUT = Path(__file__).resolve().parent
SIM = OUT.parents[1]
os.environ['MPLCONFIGDIR'] = str(OUT / 'mplconfig')
sys.dont_write_bytecode = True
sys.path[:0] = [str(SIM), str(SIM.parent)]
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import import_bootstrap
import train
from train.implementation import load_best_node_vals, session_z_from_node_vals
from train.cost import _pack_cost_parts_from_v_readout, pack_traces, gt_affine_from_pack, _weighted_mean_cost_parts, forward_pack
from neuron.forward import forward_v
import neuron.point_hh as hh
from figure.mbar import _traces_from_forward
from task.mbar.pack import bar_specs_from_task, dsi_values

torch.set_num_threads(1)
HH = next((SIM/'0_runs/point_hh').glob('34427799-*'))
HP = next((SIM/'0_runs/hp_lp').glob('34381872-*'))
HP100 = next((SIM/'0_runs/hp_lp').glob('34381255-*'))
sources = list((SIM/'1_neuron').glob('*.py')) + list((SIM/'4_train').glob('*.py')) + [SIM/'conf/config.yaml']
sources += [r/'data'/f for r in (HH, HP, HP100) for f in ('train_opts.json', 'best_param.npz')]
hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
models = {}
metrics = []
pathmetrics = []

def write_csv(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def align_subset(arr, nodes, traces, tokens, nc, cells):
    result = {}
    for b, token in enumerate(tokens):
        nt = traces.before_t[token] + traces.after_t[token] + 1
        for cell in ('T4a','T4b'):
            local = np.flatnonzero((nc[nodes] == cells.index(cell)) & (traces.t0_bn[b,nodes] >= 0))
            ns = nodes[local]
            ts = traces.t0_bn[b,ns,None] + np.arange(nt)[None,:]
            values = arr[b, np.clip(ts,0,arr.shape[1]-1), local[:,None]].copy()
            values[ts<0] = 0.0
            result[cell,token] = values.mean(0)
    return result

for label, run, dt in [('HH',HH,1.), ('HPLP_native',HP,2.), ('HPLP',HP,1.), ('HPLP100',HP100,1.)]:
    print('START', label, flush=True)
    opts = json.loads((run/'data/train_opts.json').read_text())
    if label == 'HPLP100':
        opts['mbar_sti_opts'] = json.loads((HH/'data/train_opts.json').read_text())['mbar_sti_opts']
    opts['device'] = 'cpu'
    for key in ('spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts'):
        opts[key]['delta_ms'] = dt; opts[key]['delta_ms_pre'] = dt
    session = train.resolve_session(opts)
    session,z = session_z_from_node_vals(*load_best_node_vals(str(run)),session)
    params = train.params_from_z(z,session)
    pack = session.packs['mbar']['bright']
    cells = list(session.connectome.cells)
    nc = session.connectome.node_cells.numpy()
    specs = bar_specs_from_task(session,'mbar','bright'); tokens = [s.token for s in specs]
    source = session.connectome.conn.source_idxs
    target = session.connectome.conn.target_idxs
    post_nodes = np.flatnonzero(np.isin(nc,[cells.index('T4a'),cells.index('T4b')]))
    incoming = torch.as_tensor(np.flatnonzero(np.isin(target.numpy(),post_nodes)))
    target_local = torch.as_tensor(np.searchsorted(post_nodes,target[incoming].numpy()))
    pre_cells = nc[source[incoming].numpy()]
    incoming_types = [c for c in cells if cells.index(c) in pre_cells]
    edge_by_type = {c:torch.as_tensor(np.flatnonzero(pre_cells==cells.index(c))) for c in incoming_types}
    logs = {kind:{c:[] for c in incoming_types} for kind in ('current','conductance','release')}
    original = hh.step
    if label == 'HH':
        def recorder(state, p, drive, sess, **kw):
            new = original(state,p,drive,sess,**kw)
            s = new['s'][:,incoming]
            eg = s * p['gS'][incoming]
            ei = eg * (p['e_syn'][incoming] - new['v'][:,target[incoming]])
            for kind,val in [('current',ei),('conductance',eg),('release',s)]:
                for c, ix in edge_by_type.items():
                    agg = torch.zeros(s.shape[0],len(post_nodes))
                    agg.scatter_add_(1,target_local[ix].expand(s.shape[0],-1),val[:,ix])
                    if kind == 'release':
                        cnt = torch.bincount(target_local[ix],minlength=len(post_nodes)).clamp_min(1)
                        agg /= cnt
                    logs[kind][c].append(agg.numpy())
            return new
        hh.step = recorder
    try:
        with torch.no_grad():
            v = forward_pack(session,params,pack.i_sti,pack)
            assert torch.isfinite(v).all()
            cp = _weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,session,*gt_affine_from_pack(params,pack,session),pack_traces(v,pack),trace=v),session)
            ds = {k:float(x) for k,x in dsi_values(pack,v).items()}
    finally:
        hh.step = original
    traces,*_ = _traces_from_forward(session,'mbar','bright',v.numpy(),specs,tokens)
    curves = traces.ca_mean_cell
    data = {}; rows = []
    for (c,token),y in curves.items():
        t = (np.arange(len(y))-traces.before_t[token])*dt
        base = float(y[(t>=-350)&(t<=-180)].mean())
        mask = (t>=-150)&(t<=576); late=(t>=150)&(t<=576)
        response = y-base
        row = dict(model=label,cell=c,direction=token,baseline_mV=base,peak_mV=float(response[mask].max()),trough_mV=float(response[mask].min()),late_trough_mV=float(response[late].min()),peak_time_ms=float(t[mask][response[mask].argmax()]),trough_time_ms=float(t[mask][response[mask].argmin()]))
        metrics.append(row); rows.append(row)
        data[c+'__'+token]=y;data['time__'+token]=t
    currents = {}
    if label == 'HH':
        for kind,by_type in logs.items():
            for c,values in by_type.items():
                a = np.stack(values,axis=1)
                a = np.concatenate([a[:,:1],a],axis=1)
                aligned = align_subset(a,post_nodes,traces,tokens,nc,cells)
                for (post,token),y in aligned.items():
                    t=data['time__'+token];base=float(y[(t>=-350)&(t<=-180)].mean());m=(t>=-150)&(t<=576)
                    pathmetrics.append(dict(kind=kind,source=c,target=post,direction=token,baseline=base,minimum=float(y[m].min()),maximum=float(y[m].max()),delta_min=float(y[m].min()-base),delta_max=float(y[m].max()-base)))
                    currents[kind+'__'+c+'__'+post+'__'+token]=y
        np.savez_compressed(OUT/'HH_pathway_traces.npz',**currents)
        # Per-edge data show how voltage reaches the trained release nonlinearity.
        edge_rows=[]
        for c,ix in edge_by_type.items():
            es=incoming[ix]
            edge_rows.append(dict(source=c,n_edges=len(es),v_th_syn=float(params['v_th_syn'][es].mean()),delta_syn=float(params['delta_syn'][es].mean()),k_minus_syn=float(params['k_minus_syn'][es].mean()),gS_mean_nS=float(params['gS'][es].mean()),gS_total_nS=float(params['gS'][es].sum()),e_syn_mean_mV=float(params['e_syn'][es].mean())))
        write_csv('HH_T4_edge_parameters.csv',edge_rows)
    np.savez_compressed(OUT/(label+'_aligned.npz'),**data)
    saved=np.load(run/'data/costs.npy')
    models[label]=dict(run=str(run),dt_ms=dt,cost=sum(float(x) for x in cp.values()),parts={k:float(x) for k,x in cp.items()},saved_cost=float(saved.min()),dsi_raw=ds,nodes=int(session.connectome.n_node),incoming_types=incoming_types,a_sti_mid=float(params['a_sti_mid'].reshape(-1)[0]))
    print(json.dumps(models[label]),flush=True)
    del v,session,z,params,logs,curves,traces;gc.collect()

write_csv('waveform_metrics.csv',metrics)
write_csv('HH_pathway_metrics.csv',pathmetrics)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
(OUT/'summary.json').write_text(json.dumps(dict(models=models,source_sha256=hashes,source_unchanged=True,notes=['Saved HPLP used 2 ms; common-grid replay uses 1 ms without retraining.','Cost is recomputed with current local cost code; inspect difference from saved cost.','Aligned population traces follow the existing mbar figure alignment.','Raw DSI is an unnormalized sum difference, not the standard peak DSI.']),indent=2)+'\n')

hhc=np.load(OUT/'HH_aligned.npz');hpc=np.load(OUT/'HPLP_aligned.npz')
cs=['L1','L2','Mi1','Tm3','Mi4','Mi9','T4a','T4b']
fig,axes=plt.subplots(len(cs),2,figsize=(12,18),sharex=True,sharey='row')
for i,c in enumerate(cs):
    for j,token in enumerate(tokens):
        ax=axes[i,j]
        for a,name,color in [(hhc,'HH 34427799','#cc493d'),(hpc,'HPLP 34381872','#296bb1')]:
            t=a['time__'+token];y=a[c+'__'+token];base=y[(t>=-350)&(t<=-180)].mean()
            ax.plot(t,y-base,label=name,color=color,lw=1.5)
        ax.axhline(0,color='.5',lw=.6);ax.axvline(0,color='.5',ls=':',lw=.7);ax.grid(alpha=.15);ax.set_xlim(-200,650)
        if j==0:ax.set_ylabel(c+' delta V (mV)')
        if i==0:ax.set_title(token.replace('_',' '))
        if i==len(cs)-1:ax.set_xlabel('Time from local first stimulus (ms)')
axes[0,0].legend(frameon=False,fontsize=9)
fig.suptitle('Same bright moving-bar task | both replayed at dt=1 ms\nPopulation mean, local stimulus aligned; baseline = -350 to -180 ms; no retraining',fontsize=13)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'upstream_comparison.png',dpi=150);plt.close(fig)
paths=np.load(OUT/'HH_pathway_traces.npz')
fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
for j,token in enumerate(tokens):
    t=hhc['time__'+token]
    for i,kind in enumerate(['release','conductance','current']):
        ax=axes[i,j]
        for c in models['HH']['incoming_types']:
            y=paths[kind+'__'+c+'__T4a__'+token];base=y[(t>=-350)&(t<=-180)].mean()
            ax.plot(t,y-base,label=c,lw=1.4)
        ax.axhline(0,color='.5',lw=.6);ax.set_xlim(-200,650);ax.grid(alpha=.15)
        ax.set_ylabel({'release':'Change in mean release probability','conductance':'Change in input conductance (nS)','current':'Change in inward current (pA)'}[kind])
        if i==0:ax.set_title('T4a '+token)
        if i==2:ax.set_xlabel('Time from T4 local first stimulus (ms)')
axes[0,0].legend(frameon=False,ncol=3)
fig.suptitle('HH: actual input onto T4a split by source type\nChanges relative to -350 to -180 ms baseline; current also depends on T4 voltage',fontsize=13)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'HH_T4a_inputs.png',dpi=150);plt.close(fig)
print('FINISHED',OUT,flush=True)
