"""Prepare matched moving-bar reference traces; all outputs stay in this folder."""
from pathlib import Path
import sys,os,json,hashlib,shutil,gc
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1]
sys.dont_write_bytecode=True;os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig');sys.path[:0]=[str(SIM),str(SIM.parent)]
import numpy as np
import torch
import import_bootstrap
import train
import neuron.point_hh as hh
import task.mbar.pack as mp
from train.implementation import load_best_node_vals,session_z_from_node_vals
from train.cost import forward_pack,_pack_cost_parts_from_v_readout,_weighted_mean_cost_parts,gt_affine_from_pack,pack_traces
from task.mbar.pack import dsi_values,bar_specs_from_task
from figure.mbar import _traces_from_forward
from neuron.forward import _mbar_gaussian_overlay

torch.set_num_threads(1)
HH=next((SIM/'0_runs/point_hh').glob('34427799-*'));HP=next((SIM/'0_runs/hp_lp').glob('34381872-*'))
original_signals=mp.build_mbar_signals
mp.build_mbar_signals=lambda *a,**kw:original_signals(*a,**dict(kw,use_cache=False))


def load(run):
 opts=json.loads((run/'data/train_opts.json').read_text());opts['device']='cpu'
 for key in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[key]['delta_ms']=1.;opts[key]['delta_ms_pre']=1.
 session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(run)),session);return session,z,train.params_from_z(z,session)


def cost_and_curves(session,p,pack,v):
 cp=_weighted_mean_cost_parts(_pack_cost_parts_from_v_readout(pack,session,*gt_affine_from_pack(p,pack,session),pack_traces(v,pack),trace=v),session)
 specs=bar_specs_from_task(session,'mbar','bright');tokens=[x.token for x in specs];tr,*_=_traces_from_forward(session,'mbar','bright',v.numpy(),specs,tokens);curves={};metrics={}
 for (cell,token),y in tr.ca_mean_cell.items():
  t=np.arange(len(y))-tr.before_t[token];base=float(y[(t>=-350)&(t<=-180)].mean());yy=y-base;mask=(t>=-150)&(t<=576)
  curves[cell+'__'+token]=yy;curves['time__'+token]=t;metrics[cell+'__'+token]=dict(peak=float(yy[mask].max()),trough=float(yy[mask].min()))
 dsi={k:float(a) for k,a in dsi_values(pack,v).items()}
 peaks={}
 for cell in ['T4a','T4b']:
  pd='right_bright_w1' if cell=='T4a' else 'left_bright_w1';nd='left_bright_w1' if cell=='T4a' else 'right_bright_w1';a=max(0.,metrics[cell+'__'+pd]['peak']);b=max(0.,metrics[cell+'__'+nd]['peak']);peaks[cell]=dict(PD_peak=a,ND_peak=b,peak_DSI=(a-b)/(a+b) if a+b>0 else None)
 return dict(cost=sum(float(x) for x in cp.values()),cost_parts={k:float(x) for k,x in cp.items()},dsi_raw=dsi,peak_metrics=peaks,waveform_metrics=metrics),curves


def main():
 (OUT/'source_snapshot').mkdir(exist_ok=True)
 for label,run in [('HH',HH),('HPLP',HP)]:
  d=OUT/'source_snapshot'/label;d.mkdir(exist_ok=True)
  for name in ['train_opts.json','best_param.npz']:shutil.copyfile(run/'data'/name,d/name)
 sources=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/1_2_hp_lp.py',SIM/'1_neuron/4_forward.py',SIM/'4_train/3_cost.py',SIM/'4_train/1_param.py',SIM/'4_train/2_session.py',SIM/'3_task/4_mbar/4_pack.py']
 manifest=dict(HH_run=str(HH),HPLP_run=str(HP),dt_ms=1.,source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},reference_notes='Primary HPLP reference reuses HH moving-bar current geometry and HH learned Gaussian sigma. No retraining.')
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 H,z,p=load(HH);pack=H.packs['mbar']['bright'];nc=H.connectome.node_cells;cells=list(H.connectome.cells);src=H.connectome.conn.source_idxs;tar=H.connectome.conn.target_idxs
 selected=torch.isin(nc,torch.tensor([cells.index('T4a'),cells.index('T4b')]));outgoing=selected[src];proof=dict(n_outgoing_edges=int(outgoing.sum()),max_outgoing_gS=float(p['gS'][outgoing].max()),all_outgoing_zero=bool((p['gS'][outgoing]==0).all()),n_target_nodes=int(selected.sum()),n_incoming_edges=int(selected[tar].sum()))
 (OUT/'conditional_graph_proof.json').write_text(json.dumps(proof,indent=2)+'\n');assert proof['all_outgoing_zero'],proof
 results={}
 with torch.no_grad():
  for name,k in [('HH_saved',None),('HH_k001',.01)]:
   pp={key:(v.clone() if torch.is_tensor(v) else v) for key,v in p.items()}
   if k is not None:pp['k_minus_syn'][:]=k
   print('START',name,'k=',float(pp['k_minus_syn'][0]),flush=True)
   v=forward_pack(H,pp,pack.i_sti,pack);assert torch.isfinite(v).all();np.save(OUT/(name+'_full.npy'),v.numpy())
   # Same constant background for every batch; source geometry is unchanged.
   control=forward_pack(H,pp,pack.i_sti[:,:1,:].expand_as(pack.i_sti),None if False else type('Background',(),{'t_onset':pack.t_onset})())
   np.save(OUT/(name+'_background.npy'),control.numpy());metrics,curves=cost_and_curves(H,pp,pack,v);results[name]=dict(k_minus_syn=float(pp['k_minus_syn'][0]),**metrics);np.savez_compressed(OUT/(name+'_curves.npz'),**curves)
   print('DONE',name,metrics['cost'],metrics['dsi_raw'],flush=True);del v,control;gc.collect()
  P,zp,pp=load(HP);assert list(P.connectome.node_ids)==list(H.connectome.node_ids);assert torch.equal(P.connectome.conn.source_idxs,src) and torch.equal(P.connectome.conn.target_idxs,tar)
  pp['a_sti_mid'][:]=p['a_sti_mid'];assert float(pp['a_sti_mid'][0])==float(p['a_sti_mid'][0])
  for ti in [0,pack.t_onset,pack.t_onset+100,int(pack.i_sti.shape[1])-1]:assert torch.equal(_mbar_gaussian_overlay(pack.i_sti[:,ti],p,pack,t=ti),_mbar_gaussian_overlay(pack.i_sti[:,ti],pp,pack,t=ti))
  print('START HPLP matched input',flush=True);v=forward_pack(P,pp,pack.i_sti,pack);np.save(OUT/'HPLP_matched_full.npy',v.numpy())
  control=forward_pack(P,pp,pack.i_sti[:,:1,:].expand_as(pack.i_sti),type('Background',(),{'t_onset':pack.t_onset})());np.save(OUT/'HPLP_matched_background.npy',control.numpy());metrics,curves=cost_and_curves(P,pp,pack,v);results['HPLP_matched']=metrics;np.savez_compressed(OUT/'HPLP_matched_curves.npz',**curves)
  print('DONE HPLP',metrics['cost'],metrics['dsi_raw'],flush=True)
 (OUT/'references.json').write_text(json.dumps(results,indent=2)+'\n');print('PREPARE_FINISHED',flush=True)
if __name__=='__main__':main()
