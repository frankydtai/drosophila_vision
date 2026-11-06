from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from scipy.special import expit
from scipy.sparse import coo_matrix,diags,bmat,csr_matrix
from scipy.sparse.linalg import spsolve,eigs,ArpackNoConvergence
SIM=Path('/home/yitai/drosophila_vision/vision/simulation');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(SIM),str(SIM.parent)]
import import_bootstrap
import train
from train.implementation import load_best_node_vals,session_z_from_node_vals
from neuron import point_hh
RUN=next((SIM/'0_runs/point_hh').glob('34400581-*'))
torch.set_num_threads(1)
def load():
 opts=json.loads((RUN/'data/train_opts.json').read_text());opts.update(tasks=['spread'],contrasts=['bright'],device='cpu')
 opts['spread_sti_opts']['delta_ms']=1.;opts['spread_sti_opts']['delta_ms_pre']=1.
 session=train.resolve_session(opts);session,z=session_z_from_node_vals(*load_best_node_vals(str(RUN)),session);p=train.params_from_z(z,session)
 pp={k:(v.detach().to(torch.float64) if v.is_floating_point() else v.detach()) for k,v in p.items() if torch.is_tensor(v)}
 I=session.packs['spread']['bright'].i_sti[0,0].to(torch.float64)
 with torch.no_grad():st=point_hh.pre_steady(session,pp,1,i_sti=I[None,None,:])
 return session,pp,I,st

def rates(v):
 def trap(x,s):
  z=x/s;small=np.abs(z)<1e-7;safe=np.where(small,1,z)
  return np.where(small,s*(1-z/2+z*z/12),x/np.expm1(safe))
 return (.1*trap(-(v+40),10),4*np.exp(-(v+65)/18),.07*np.exp(-(v+65)/20),1/(np.exp(-(v+35)/10)+1),.01*trap(-(v+55),10),.125*np.exp(-(v+65)/80))
def gates(v):
 am,bm,ah,bh,an,bn=rates(v);return am/(am+bm),ah/(ah+bh),an/(an+bn)
def main():
 session,p,I,st=load();P={k:v.numpy() for k,v in p.items()};I=I.numpy();N=session.connectome.n_node;cells=list(session.connectome.cells);nc=session.connectome.node_cells.numpy()
 source=session.connectome.conn.source_idxs.numpy();target=session.connectome.conn.target_idxs.numpy();active=P['gS']>0;src=source[active];tar=target[active];gs=P['gS'][active];er=P['e_syn'][active];th=P['v_th_syn'][active];de=P['delta_syn'][active];E=len(src)
 C=P['cap'];gna=P['gNa'];gk=P['gK'];gl=P['gLeak'];ena=P['eNa'];ek=P['eK'];el=P['eLeak']
 def quantities(v):
  m,h,n=gates(v);dmg,dhg,dng=[np.imag(x)/1e-20 for x in gates(v.astype(complex)+1e-20j)]
  ss=expit((v[src]-th)/de);dss=ss*(1-ss)/de;gan=gna*m**3*h;gkn=gk*n**4
  synG=np.bincount(tar,weights=gs*ss,minlength=N);synGE=np.bincount(tar,weights=gs*ss*er,minlength=N);G=gan+gkn+gl+synG
  F=I+gan*ena+gkn*ek+gl*el+synGE-G*v
  D=-G-gna*(3*m*m*dmg*h+m**3*dhg)*(v-ena)-gk*4*n**3*dng*(v-ek)
  Jdc=diags(D)+coo_matrix((gs*dss*(er-v[tar]),(tar,src)),shape=(N,N)).tocsr()
  return F,Jdc,(m,h,n,dmg,dhg,dng,ss,dss,G)
 v=st['v'][0].numpy().copy();initial=v.copy();history=[]
 for it in range(80):
  F,J,Q=quantities(v);norm=float(np.linalg.norm(F));maximum=float(abs(F).max());history.append(dict(iter=it,l2_pA=norm,max_pA=maximum));print('ROOT',it,norm,maximum,flush=True)
  if maximum<1e-9:break
  dv=spsolve(J,-F);alpha=1.
  for ls in range(30):
   trial=v+alpha*dv
   if np.min(trial)>-119. and np.max(trial)<79.:
    Fn,_,_=quantities(trial)
    if np.linalg.norm(Fn)<norm:break
   alpha*=.5
  else:raise RuntimeError('Newton line search failed')
  v=trial
 else:raise RuntimeError('No DC root')
 F,Jdc,Q=quantities(v);m,h,n,dm,dh,dn,ss,dss,G=Q
 # Numerical directional check for the DC Jacobian.
 rng=np.random.default_rng(421);direction=rng.normal(size=N);eps=1e-5;fd=(quantities(v+eps*direction)[0]-quantities(v-eps*direction)[0])/(2*eps);jd=Jdc@direction
 check=float(np.linalg.norm(fd-jd)/np.linalg.norm(fd));assert check<1e-7,check
 am,bm,ah,bh,an,bn=rates(v);qrate=np.r_[am+bm,ah+bh,an+bn];qprime=np.r_[dm,dh,dn]
 B=bmat([[diags(-gna*3*m*m*h*(v-ena)/C),diags(-gna*m**3*(v-ena)/C),diags(-gk*4*n**3*(v-ek)/C)]],format='csr')
 S=coo_matrix((gs*(er-v[tar])/C[tar],(tar,np.arange(E))),shape=(N,E)).tocsr()
 Qv=bmat([[diags(dm)],[diags(dh)],[diags(dn)]],format='csr')
 Sv=coo_matrix((dss,(np.arange(E),src)),shape=(E,N)).tocsr()
 results=dict(n_node=N,n_active_edge=E,max_dc_residual_pA=float(abs(F).max()),voltage_min_mV=float(v.min()),voltage_max_mV=float(v.max()),dc_jacobian_relative_error=check,initial_presteady_max_residual_pA=history[0]['max_pA'],root_history=history,models={})
 np.savez_compressed(OUT/'dc_equilibrium.npz',v=v,m=m,h=h,n=n,s_active=ss,active=active,initial_v=initial,G=G,**{k:val for k,val in P.items()})
 for km in [.1,.01]:
  r=km/(1-ss)
  J=bmat([[diags(-G/C),B,S],[diags(qrate)@Qv,diags(-qrate),None],[diags(r)@Sv,None,diags(-r)]],format='csc')
  info={'continuous':[],'discrete':{}}
  # Shift-invert locates modes near the observed 100-150 Hz fluctuations.
  for sigma in [0.,.1+.75j]:
   print('EIG continuous',km,sigma,'shape',J.shape,'nnz',J.nnz,flush=True)
   try:
    val,vec=eigs(J,k=10,sigma=sigma,which='LM',tol=1e-9,maxiter=1500)
   except ArpackNoConvergence as exc:
    val,vec=exc.eigenvalues,exc.eigenvectors
   for k,value in enumerate(val):
    participation=np.bincount(nc,weights=abs(vec[:N,k])**2,minlength=len(cells));order=np.argsort(participation)[::-1][:6]
    info['continuous'].append(dict(real=float(value.real),imag=float(value.imag),frequency_Hz=float(abs(value.imag)*1000/(2*np.pi)),cells={cells[i]:float(participation[i]/participation.sum()) for i in order}))
   print('VALUES',val,flush=True)
  for dt in [1.,.1,.025,.00625]:
   Rq=np.exp(-dt*qrate);Rs=np.exp(-dt*r);U=diags(1-Rq)@Qv;W=diags(1-Rs)@Sv;T=diags(dt*C/(C+dt*G));Mv=diags(C/(C+dt*G))
   A=bmat([[Mv+T@(B@U+S@W),T@B@diags(Rq),T@S@diags(Rs)],[U,diags(Rq),None],[W,None,diags(Rs)]],format='csr')
   print('EIG discrete',km,dt,flush=True)
   try:val=eigs(A,k=8,which='LM',tol=1e-8,maxiter=3000,return_eigenvectors=False)
   except ArpackNoConvergence as exc:val=exc.eigenvalues
   info['discrete'][str(dt)]=[dict(real=float(x.real),imag=float(x.imag),modulus=float(abs(x)),growth_per_ms=float(np.log(abs(x))/dt),frequency_Hz=float(abs(np.angle(x))*1000/(2*np.pi*dt))) for x in val]
   print('DISCRETE_VALUES',val,flush=True)
  results['models'][str(km)]=info;(OUT/'stability_summary.json').write_text(json.dumps(results,indent=2)+'\n')
 print('DONE',flush=True)
if __name__=='__main__':main()
