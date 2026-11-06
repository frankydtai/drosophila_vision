from pathlib import Path
import json
import numpy as np
from scipy.sparse import diags,coo_matrix,bmat
from scipy.sparse.linalg import eigs
from stability import load,rates,gates,OUT
session,p,I,initial=load();z=np.load(OUT/'dc_equilibrium.npz');v=z['v'];m=z['m'];h=z['h'];n=z['n'];G=z['G'];P={k:val.numpy() for k,val in p.items()};N=len(v);active=P['gS']>0;src=session.connectome.conn.source_idxs.numpy()[active];tar=session.connectome.conn.target_idxs.numpy()[active];gs=P['gS'][active];er=P['e_syn'][active];C=P['cap'];E=len(src)
ss=z['s_active'];dss=ss*(1-ss)/P['delta_syn'][active];dm,dh,dn=[np.imag(x)/1e-20 for x in gates(v.astype(complex)+1e-20j)];am,bm,ah,bh,an,bn=rates(v);qr=np.r_[am+bm,ah+bh,an+bn]
bm=-P['gNa']*3*m*m*h*(v-P['eNa'])/C;bh=-P['gNa']*m**3*(v-P['eNa'])/C;bn=-P['gK']*4*n**3*(v-P['eK'])/C
B=bmat([[diags(bm),diags(bh),diags(bn)]],format='csr');S=coo_matrix((gs*(er-v[tar])/C[tar],(tar,np.arange(E))),shape=(N,E)).tocsr();Qv=bmat([[diags(dm)],[diags(dh)],[diags(dn)]],format='csr');Sv=coo_matrix((dss,(np.arange(E),src)),shape=(E,N)).tocsr();nc=session.connectome.node_cells.numpy();cells=list(session.connectome.cells)
# With all synaptic release fixed at its exact equilibrium, each neuron is a 4x4 independent HH system.
local=np.zeros((N,4,4));local[:,0,0]=-G/C;local[:,0,1]=bm;local[:,0,2]=bh;local[:,0,3]=bn
for j,(r,qp) in enumerate([(am+rates(v)[1],dm),(ah+rates(v)[3],dh),(an+rates(v)[5],dn)]):local[:,j+1,0]=r*qp;local[:,j+1,j+1]=-r
local_eigs=np.linalg.eigvals(local);results={'fixed_synapses_max_real_per_ms':float(local_eigs.real.max()),'fixed_synapses_unstable_neurons':int((local_eigs.real.max(1)>0).sum()),'models':{}}
for km,targets in [(.1,[.025+.3j,.025+.8j]),(.01,[.0025+.03j,.0025+.1j])]:
 r=km/(1-ss);J=bmat([[diags(-G/C),B,S],[diags(qr)@Qv,diags(-qr),None],[diags(r)@Sv,None,diags(-r)]],format='csc').astype(complex)
 model={'continuous_modes':[]}
 for sigma in targets:
  print('START',km,sigma,flush=True);w,U=eigs(J,k=6,sigma=sigma,which='LM',tol=1e-10,maxiter=2000)
  for i,x in enumerate(w):
   en=np.bincount(nc,weights=abs(U[:N,i])**2,minlength=len(cells));idx=np.argsort(en)[::-1]
   row=dict(real_per_ms=float(x.real),imag_per_ms=float(x.imag),frequency_Hz=float(abs(x.imag)*1000/(2*np.pi)),efold_growth_ms=float(1/x.real) if x.real>0 else None,relative_residual=float(np.linalg.norm(J@U[:,i]-x*U[:,i])/np.linalg.norm(U[:,i])),voltage_participation={cells[j]:float(en[j]/en.sum()) for j in idx[:8]})
   model['continuous_modes'].append(row)
  print('VALUES',w,flush=True)
  j=int(np.argmax(w.real))
  if w[j].real>0:np.savez_compressed(OUT/f'unstable_mode_k_{km:g}_sigma_{abs(sigma.imag):g}.npz',eigenvalue=w[j],eigenvector=U[:,j],source=src,target=tar,active=active)
 # Freeze membrane gates, leaving only voltage and release feedback.
 Jsyn=bmat([[diags(-G/C),S],[diags(r)@Sv,diags(-r)]],format='csc').astype(complex)
 sigma=targets[0];print('START freeze membrane gates',km,flush=True);w=eigs(Jsyn,k=6,sigma=sigma,which='LM',tol=1e-9,maxiter=2000,return_eigenvectors=False)
 model['fixed_membrane_gate_modes']=[dict(real_per_ms=float(x.real),imag_per_ms=float(x.imag)) for x in w];print('GATE_FROZEN_VALUES',w,flush=True)
 results['models'][str(km)]=model;(OUT/'mode_summary.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
