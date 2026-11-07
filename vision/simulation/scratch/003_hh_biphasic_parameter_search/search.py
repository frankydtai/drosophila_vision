"""Fixed-1-ms Point-HH parameter search; runtime overrides only."""
from pathlib import Path
from itertools import product
import sys,os,json,csv,hashlib,time,gc
OUT=Path(__file__).resolve().parent;SIM=OUT.parents[1]
sys.dont_write_bytecode=True;os.environ['MPLCONFIGDIR']=str(OUT/'mplconfig');sys.path[:0]=[str(SIM),str(SIM.parent)]
import numpy as np
import torch
import import_bootstrap
import train
from neuron import point_hh as hh
from train.implementation import load_best_node_vals,session_z_from_node_vals

torch.set_num_threads(1)
RUN=next((SIM/'0_runs/point_hh').glob('34427799-*'))
opts=json.loads((RUN/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
for k in ['spread_sti_opts','spot_sti_opts','mbar_sti_opts','sbar_sti_opts']:opts[k]['delta_ms']=1.;opts[k]['delta_ms_pre']=1.
SESSION=train.resolve_session(opts);SESSION,Z=session_z_from_node_vals(*load_best_node_vals(str(RUN)),SESSION)
BASE=train.params_from_z(Z,SESSION);BASE['k_minus_syn']=torch.full_like(BASE['k_minus_syn'],.01)
CON=SESSION.connectome;NC=CON.node_cells;CELLS=list(CON.cells);SRC=CON.conn.source_idxs;TAR=CON.conn.target_idxs
COUNTS=torch.bincount(NC,minlength=len(CELLS));P0=SESSION.packs['spread']['bright'];I0=P0.i_sti[:,0].clone();ILIGHT=P0.i_sti[:,P0.t_onset+10].clone()
assert not torch.equal(I0,ILIGHT)
EDGE_PRE=NC[SRC];EDGE_POST=NC[TAR];RELAYS=['Mi1','Tm3'];L1=CELLS.index('L1')
FORWARD=(EDGE_PRE==L1)&torch.isin(EDGE_POST,torch.tensor([CELLS.index(c) for c in RELAYS]))
FEEDBACK=torch.isin(EDGE_PRE,torch.tensor([CELLS.index(c) for c in RELAYS]))&(EDGE_POST==L1)
FOCUS=torch.isin(EDGE_PRE,torch.tensor([CELLS.index(c) for c in ['L1','Mi1','Tm3']]))
node_reference=BASE.get('bias_gt');REST=torch.tensor([float(node_reference[c]) for c in range(len(CELLS))])
COLUMNS=hh._MEMBRANE_PARAM_NAMES+hh._SYNAPSE_PARAM_NAMES
HASHFILES=[SIM/'1_neuron/1_4_point_hh.py',SIM/'1_neuron/4_forward.py',SIM/'conf/config.yaml',RUN/'data/best_param.npz',RUN/'data/train_opts.json']
HASHES={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HASHFILES}

def make(case):
 p={k:BASE[k].clone() for k in COLUMNS}
 p['gS'][FORWARD]*=case.get('forward_gain',1.)
 p['gS'][FEEDBACK]*=case.get('feedback_gain',1.)
 mode=case.get('threshold','saved')
 if mode!='saved':
  offset={'rest':0.,'restminus4':-4.,'restplus4':4.}[mode]
  p['v_th_syn'][FOCUS]=REST[EDGE_PRE[FOCUS]]+offset
 return p

def batched(cases):
 copies=[make(c) for c in cases];p={k:torch.stack([q[k] for q in copies for _ in range(2)]) for k in COLUMNS}
 p.update(_point_hh_prepared=True,_source_idx=SRC,_target_idx=TAR)
 for k in ['cap','gNa','gK','gLeak','gS','delta_syn','k_minus_syn']:assert torch.isfinite(p[k]).all() and (p[k]>0).all() if k not in ['gS'] else torch.isfinite(p[k]).all() and (p[k]>=0).all()
 assert torch.all(p['k_minus_syn']==.01)
 return p

@torch.no_grad()
def equivalence():
 cases=[dict(name='reference'),dict(name='nontrivial',forward_gain=.3,feedback_gain=3.,threshold='rest')]
 p=batched(cases);b=len(cases)*2;i=I0.expand(b,-1)
 state=hh.pre_steady(SESSION,p,b,i_sti=i.unsqueeze(1))
 originals=[]
 for case in cases:
  pp=make(case);pp=hh.prepare_params(pp,I0,SESSION);ss=hh.pre_steady(SESSION,pp,1,i_sti=I0.unsqueeze(1));originals.append((ss,pp))
 max_error=0.
 for step in range(101):
  for j,(ss,pp) in enumerate(originals):
   for k in hh.STATE_KEYS:
    max_error=max(max_error,float((state[k][2*j]-ss[k][0]).abs().max()))
   assert torch.allclose(state['v'][2*j],state['v'][2*j+1],atol=1e-5,rtol=1e-5)
  if step==100:break
  drive=(ILIGHT if step>=40 else I0)
  state=hh.step(state,p,drive.expand(b,-1),SESSION,delta_ms=1.)
  originals=[(hh.step(ss,pp,drive,SESSION,delta_ms=1.),pp) for ss,pp in originals]
 assert max_error<2e-4,max_error
 return max_error

@torch.no_grad()
def run_batch(cases,warmup=1000,post=1000,duration=160):
 b=len(cases)*2;p=batched(cases);state=hh.pre_steady(SESSION,p,b,i_sti=I0.expand(b,-1).unsqueeze(1))
 total=warmup+duration+post+1;means=np.empty((b,total,len(CELLS)),dtype=np.float32);selected={c:[] for c in ['L1','L2','Mi1','Tm3','Mi9']}
 vmin=1e9;vmax=-1e9
 # The pulse and its matched constant-background control share the same initial state.
 drive0=I0.expand(b,-1);driveon=drive0.clone();driveon[::2]=ILIGHT
 for ti in range(total):
  v=state['v'];avg=torch.zeros((b,len(CELLS)));avg.scatter_add_(1,NC.expand(b,-1),v);avg/=COUNTS;means[:,ti]=avg.numpy()
  if ti%100==0:
   assert all(torch.isfinite(a).all() for a in state.values());vmin=min(vmin,float(v.min()));vmax=max(vmax,float(v.max()))
  if ti==total-1:break
  drive=driveon if warmup<=ti<warmup+duration else drive0
  state=hh.step(state,p,drive,SESSION,delta_ms=1.)
 assert all(torch.isfinite(a).all() for a in state.values())
 assert all(((state[k]>=0)&(state[k]<=1)).all() for k in ['m','h','n','s'])
 t=np.arange(total)-warmup;records=[]
 for j,case in enumerate(cases):
  yp=means[2*j];yc=means[2*j+1];dy=yp-yc;cells={}
  for ci,cell in enumerate(CELLS):
   y=dy[:,ci];on=(t>=0)&(t<duration);off=(t>=duration)&(t<=duration+700);late=t>=duration+post-150;quiet=(t>=-200)&(t<=0)
   sign=-1 if cell in ['L1','L2','L3','Mi9'] else 1
   onamp=float((sign*y[on]).max());opposite=np.maximum(-sign*y[off],0);ot=t[off];pk=int(opposite.argmax());amp=float(opposite[pk]);half=opposite>=amp/2;lo=hi=pk
   while lo>0 and half[lo-1]:lo-=1
   while hi+1<len(half) and half[hi+1]:hi+=1
   width=hi-lo+1 if amp>0 else 0
   control_std=float(yc[t>=0,ci].std());late_err=float(np.max(abs(y[late])))
   r=dict(on_peak_mV=onamp,opposite_peak_mV=amp,opposite_delay_ms=float(ot[pk]-duration),opposite_fwhm_ms=width,control_std_mV=control_std,control_late_std_mV=float(yc[late,ci].std()),late_evoked_max_mV=late_err,baseline_mV=float(yc[quiet,ci].mean()))
   cells[cell]=r
  target=[cells[c] for c in ['Mi1','Tm3']]
  strength=sum(min(r['on_peak_mV'],r['opposite_peak_mV'])*min(1.,r['opposite_fwhm_ms']/40) for r in target)/2
  quiet_penalty=1+sum(r['control_late_std_mV'] for r in target)
  decay_penalty=1+max(0.,max(r['late_evoked_max_mV'] for r in target)-1)
  score=strength/quiet_penalty/decay_penalty
  record=dict(case=case,score=score,cells=cells,v_min=vmin,v_max=vmax)
  records.append(record)
  np.savez_compressed(OUT/'curves'/f"{case['name']}.npz",time_ms=t,pulse=yp,control=yc,evoked=dy,cells=np.array(CELLS))
 return records

def main():
 (OUT/'curves').mkdir(exist_ok=True)
 print('BATCH_EQUIVALENCE',equivalence(),flush=True)
 cases=[dict(name='reference',forward_gain=1.,feedback_gain=1.,threshold='saved')]
 for i,(fg,bg,th) in enumerate(product([.3,1.,3.],[.3,1.,3.,10.],['saved','rest','restminus4','restplus4'])):
  cases.append(dict(name=f'feedback_grid_{i:03}',forward_gain=fg,feedback_gain=bg,threshold=th))
 records=[];start=time.perf_counter()
 for i in range(0,len(cases),6):
  print('START BATCH',i,'/',len(cases),flush=True)
  rr=run_batch(cases[i:i+6]);records.extend(rr)
  (OUT/'search_results.json').write_text(json.dumps(records,indent=2)+'\n')
  print('DONE BATCH',i,'elapsed',time.perf_counter()-start,'top',[(x['case'],round(x['score'],3)) for x in sorted(records,key=lambda q:q['score'],reverse=True)[:3]],flush=True)
  gc.collect()
 manifest=dict(source=str(RUN),dt_ms=1.,k_minus_syn=.01,experiment='Existing conductances and release thresholds only; native HH gate rates unchanged; full 5789-node network; paired background control; no retraining.',source_sha256=HASHES,source_unchanged={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items()},n_cases=len(cases),batch_equivalence_max_error=equivalence())
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('FINISHED',flush=True)
if __name__=='__main__':main()
