import json
import numpy as np
from stability import load,OUT
session,p,I,st=load();q=np.load(OUT/'dc_equilibrium.npz');active=q['active'];src=session.connectome.conn.source_idxs.numpy()[active];tar=session.connectome.conn.target_idxs.numpy()[active];gs=q['gS'][active];er=q['e_syn'][active];ss=q['s_active'];de=q['delta_syn'][active];v=q['v'];C=q['cap'];a=q['G']/C;nc=session.connectome.node_cells.numpy();cells=list(session.connectome.cells);lookup={(int(s),int(t)):e for e,(s,t) in enumerate(zip(src,tar))};pairs=[]
for e,(s,t) in enumerate(zip(src,tar)):
 if s<t and (int(t),int(s)) in lookup:pairs.append((e,lookup[int(t),int(s)]))
mat=np.zeros((len(pairs),4,4));rows=[]
for z,(e,f) in enumerate(pairs):
 j=int(src[e]);i=int(tar[e]);b1=gs[e]*(er[e]-v[i])/C[i];b2=gs[f]*(er[f]-v[j])/C[j];c1=.1*ss[e]/de[e];c2=.1*ss[f]/de[f];r1=.1/(1-ss[e]);r2=.1/(1-ss[f]);mat[z]=[[-a[i],0,b1,0],[0,-a[j],0,b2],[0,c1,-r1,0],[c2,0,0,-r2]]
 rows.append(dict(node_i=i,node_j=j,type_i=cells[nc[i]],type_j=cells[nc[j]],a_i=float(a[i]),a_j=float(a[j]),b1=float(b1),b2=float(b2),c1=float(c1),c2=float(c2),r1=float(r1),r2=float(r2),dc_loop_gain=float(b1*b2*c1*c2/(a[i]*a[j]*r1*r2))))
w=np.linalg.eigvals(mat)
for r,ww in zip(rows,w):
 z=ww[np.argmax(ww.real)];r.update(max_real=float(z.real),imag=float(z.imag))
rows.sort(key=lambda r:r['max_real'],reverse=True);result=dict(n_bidirectional_pairs=len(rows),unstable_pairs=sum(r['max_real']>0 for r in rows),top=rows[:10]);(OUT/'pair_loops.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
