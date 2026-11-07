"""Transient pathway controls. No source or saved model files are edited."""
from pathlib import Path
import sys, os, json, hashlib, csv, gc
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1]
sys.dont_write_bytecode=True
os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig')
sys.path[:0]=[str(SIM),str(SIM.parent)]
import numpy as np
import torch
import import_bootstrap
import train
import neuron.point_hh as hh
import neuron.hp_lp as hp
from train.implementation import load_best_node_vals,session_z_from_node_vals
from train.cost import forward_pack,_pack_cost_parts_from_v_readout,_weighted_mean_cost_parts,gt_affine_from_pack,pack_traces
from task.mbar.pack import dsi_values,bar_specs_from_task
from figure.mbar import _traces_from_forward
torch.set_num_threads(1)
HH=next((SIM/'0_runs/point_hh').glob('34427799-*'))
HP=next((SIM/'0_runs/hp_lp').glob('34381872-*'))
sources=list((SIM/'1_neuron').glob('*.py'))+list((SIM/'4_train').glob('*.py'))+[SIM/'conf/config.yaml']
sources += [r/'data'/f for r in (HH,HP) for f in ('best_param.npz','train_opts.json')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
results=[];curves={};all_pathways={}

for label,run,cases in [('HPLP',HP,['baseline','Mi9_frozen','Mi4_frozen','Mi1_Tm3_no_undershoot','baseline_repeat']),('HH',HH,['baseline','Mi9_frozen','Mi4_frozen'])]:
    opts=json.loads((run/'data/train_opts.json').read_text());opts['device']='cpu'
    for k in ('spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts'):
        opts[k]['delta_ms']=1.;opts[k]['delta_ms_pre']=1.
    sess=train.resolve_session(opts);sess,z=session_z_from_node_vals(*load_best_node_vals(str(run)),sess)
    pack=sess.packs['mbar']['bright'];conn=sess.connectome.conn;cells=list(sess.connectome.cells)
    nc=sess.connectome.node_cells;src=conn.source_idxs;tar=conn.target_idxs
    post=np.isin(nc[tar].numpy(),[cells.index('T4a'),cells.index('T4b')])
    specs=bar_specs_from_task(sess,'mbar','bright');tokens=[s.token for s in specs]
    for case in cases:
        print('START',label,case,flush=True)
        p=train.params_from_z(z,sess)
        chosen=['Mi1','Tm3'] if case=='Mi1_Tm3_no_undershoot' else [case.split('_')[0]] if 'frozen' in case else []
        indices=torch.as_tensor(np.flatnonzero(post & np.isin(nc[src].numpy(),[cells.index(c) for c in chosen])))
        old_hh_step=hh.step;old_release=hh._release_step;old_hp_step=hp.step;old_signed=conn.signed_g
        clock={'t':0,'active':False,'reference':None}
        path_logs={c:[] for c in ['Mi1','Tm3','Mi4','Mi9']}
        path_edges={c:torch.as_tensor(np.flatnonzero(post & (nc[src].numpy()==cells.index(c)))) for c in path_logs}
        def hp_step(vslow,v,params,drive,session,**kw):
            clock['t']+=1
            if clock['t']==int(pack.t_onset):
                clock['reference']=(torch.relu(v-params['v_th'])*params['a_out']).clone()
                clock['active']=True
            return old_hp_step(vslow,v,params,drive,session,**kw)
        def signed(vout,strength):
            out=old_signed(vout,strength)
            edge_strength=conn._edge_syn_strength(strength)
            if case=='baseline' and clock['t']>0:
                for c,es in path_edges.items():
                    val=vout[:,src[es]]*conn.w_signed[es]*edge_strength[es]
                    agg=torch.zeros_like(vout);agg.scatter_add_(1,tar[es].expand(vout.shape[0],-1),val)
                    path_logs[c].append((agg*p['a_in']).numpy())
            if clock['active'] and len(indices):
                delta=clock['reference'][:,src[indices]]-vout[:,src[indices]]
                if case=='Mi1_Tm3_no_undershoot':delta=delta.clamp_min(0.)
                correction=torch.zeros_like(out)
                correction.scatter_add_(1,tar[indices].expand(vout.shape[0],-1),delta*conn.w_signed[indices]*edge_strength[indices])
                out=out+correction
            return out
        def hh_step(state,params,drive,session,**kw):
            clock['t']+=1
            if clock['t']==int(pack.t_onset):
                clock['reference']=state['s'].clone();clock['active']=True
            return old_hh_step(state,params,drive,session,**kw)
        def release(*args,**kw):
            value=old_release(*args,**kw)
            if clock['active'] and len(indices):
                value=value.clone();value[:,indices]=clock['reference'][:,indices]
            return value
        if label=='HPLP':hp.step=hp_step;conn.signed_g=signed
        else:hh.step=hh_step;hh._release_step=release
        try:
            with torch.no_grad():
                v=forward_pack(sess,p,pack.i_sti,pack)
                assert torch.isfinite(v).all()
                cp=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,sess,*gt_affine_from_pack(p,pack,sess),pack_traces(v,pack),trace=v),sess)
                ds={k:float(x) for k,x in dsi_values(pack,v).items()}
        finally:
            hh.step=old_hh_step;hh._release_step=old_release;hp.step=old_hp_step;conn.signed_g=old_signed
        tr,*_=_traces_from_forward(sess,'mbar','bright',v.numpy(),specs,tokens)
        row=dict(model=label,case=case,cost=sum(float(x) for x in cp.values()),**ds)
        saved={}
        for (c,token),y in tr.ca_mean_cell.items():
            if c not in ['T4a','T4b']:continue
            t=np.arange(len(y))-tr.before_t[token];base=y[(t>=-350)&(t<=-180)].mean();d=y-base
            saved[c+'__'+token]=d;saved['time__'+token]=t
            m=(t>=-150)&(t<=576);row[c+'__'+token+'__peak']=float(d[m].max());row[c+'__'+token+'__trough']=float(d[m].min())
        curves[label+'__'+case]=saved
        if label=='HPLP' and case=='baseline':
            for c,vals in path_logs.items():
                a=np.stack(vals,axis=1);a=np.concatenate([a[:,:1],a],axis=1)
                assert a.shape[:2]==v.shape[:2]
                ptr,*_=_traces_from_forward(sess,'mbar','bright',a,specs,tokens)
                for (target,token),y in ptr.ca_mean_cell.items():
                    if target in ['T4a','T4b']:all_pathways[c+'__'+target+'__'+token]=y
            np.savez_compressed(OUT/'HPLP_input_contributions.npz',**all_pathways)
        print(json.dumps(row),flush=True);results.append(row)
        del v,tr;gc.collect()
    if label=='HPLP':
        for k in curves['HPLP__baseline']:
            assert np.array_equal(curves['HPLP__baseline'][k],curves['HPLP__baseline_repeat'][k]),k
    del sess,z;gc.collect()

assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
for name,data in curves.items():np.savez_compressed(OUT/(name+'.npz'),**data)
with (OUT/'controls.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
(OUT/'summary.json').write_text(json.dumps(dict(results=results,dt_ms=1.,source_sha256=hashes,source_unchanged=True,hplp_repeat_exact=True,method='After global stimulus onset, freeze only the specified source output onto T4a/b at its immediately preceding background value. Mi1_Tm3_no_undershoot removes only below-background output onto T4a/b. No retraining. Every runtime hook is restored in finally.'),indent=2)+'\n')
print('FINISHED',OUT,flush=True)
