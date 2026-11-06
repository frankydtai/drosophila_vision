from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from stability import OUT,rates
z=np.load(OUT/'dc_equilibrium.npz');q=np.load(OUT/'unstable_mode_k_0.1_sigma_0.3.npz');active=q['active'];src=q['source'];tar=q['target'];N=len(z['v']);gs=z['gS'][active];er=z['e_syn'][active];th=z['v_th_syn'][active];de=z['delta_syn'][active];E=len(src)
# Recover the constant drive from the already solved equilibrium currents.
v=z['v'];m=z['m'];h=z['h'];n=z['n'];s=z['s_active'];gna=z['gNa'];gk=z['gK'];gl=z['gLeak'];cap=z['cap'];ena=z['eNa'];ek=z['eK'];el=z['eLeak']
Isyn=np.bincount(tar,weights=gs*s*(er-v[tar]),minlength=N);I=gna*m**3*h*(v-ena)+gk*n**4*(v-ek)+gl*(v-el)-Isyn
# The current agrees with the actual stimulus mask: 20 pA for photoreceptors, zero otherwise.
assert np.max(np.minimum(abs(I),abs(I-20)))<1e-8
x0=np.r_[v,m,h,n,s];u=q['eigenvector'];u=u/max(abs(u[:N]));lam=complex(q['eigenvalue'])
def rhs(x):
 v=x[:N];m=x[N:2*N];h=x[2*N:3*N];n=x[3*N:4*N];s=x[4*N:];am,bm,ah,bh,an,bn=rates(v)
 syn=np.bincount(tar,weights=gs*s*(er-v[tar]),minlength=N)
 dv=(I-gna*m**3*h*(v-ena)-gk*n**4*(v-ek)-gl*(v-el)+syn)/cap
 si=expit((v[src]-th)/de);ds=.1*(si-s)/(1-si)
 return np.r_[dv,am*(1-m)-bm*m,ah*(1-h)-bh*h,an*(1-n)-bn*n,ds]
checks=[]
for component in ['real','imag']:
 direction=u.real if component=='real' else u.imag
 expected=(lam*u).real if component=='real' else (lam*u).imag
 for eps in [1e-3,1e-4,1e-5]:
  fd=(rhs(x0+eps*direction)-rhs(x0-eps*direction))/(2*eps);err=np.linalg.norm(fd-expected)/np.linalg.norm(expected);checks.append(dict(component=component,perturb_mV=eps,relative_rhs_derivative_error=float(err)))
  assert err<1e-5,checks[-1]
result=dict(equilibrium_rhs_max=float(abs(rhs(x0)).max()),drive_error_to_0_or_20_pA=float(np.max(np.minimum(abs(I),abs(I-20)))),checks=checks)
(OUT/'rhs_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
