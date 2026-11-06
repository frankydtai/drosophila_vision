from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from stability import OUT,rates,gates
fine=OUT/'constant_light_perturb_dt_0.00625.npz';r=np.load(fine if fine.exists() else OUT/'constant_light_perturb_dt_0.025.npz');dt=.00625 if fine.exists() else .025
summary=json.load(open(OUT/('demonstration_summary_000625.json' if fine.exists() else 'demonstration_summary.json')));modes=json.load(open(OUT/'mode_summary.json'));spectrum=json.load(open(OUT/'stability_summary.json'));z=np.load(OUT/'dc_equilibrium.npz')
t=r['time_ms'];y=r['perturbed_v_mV'];frozen=r['fixed_synapse_v_mV'];analytic=r['linear_prediction_mV'];growth=summary['predicted_growth_per_ms']
fig,axes=plt.subplots(2,2,figsize=(12,8.2))
ax=axes[0,0];ax.plot(t,y,color='#c43a32',lw=1.5,label=f'Full network, dt={dt:g} ms');ax.plot(t,analytic,'k--',lw=1.2,label='Continuous linear prediction');ax.plot(t,frozen,color='#2877b6',lw=1.8,label='Synaptic release fixed at equilibrium');ax.axhline(0,color='gray',lw=.5);ax.set_title('A. Constant light + 0.0001 mV initial perturbation');ax.set_xlabel('Time after perturbation (ms)');ax.set_ylabel('Mi1 voltage perturbation (mV)');ax.legend(fontsize=7);ax.set_xlim(0,280)
ax=axes[0,1];peaks,_=find_peaks(abs(y));peaks=peaks[(t[peaks]>=20)&(t[peaks]<=260)];ax.semilogy(t[peaks],abs(y[peaks]),'o',color='#c43a32',ms=4,label='Observed oscillation amplitude');amp=abs(analytic[0]);ax.semilogy(t,amp*np.exp(growth*t),'k--',label=f'Predicted envelope: exp({growth:.4f} t)');ax.semilogy(t,np.maximum(abs(frozen),1e-13),color='#2877b6',label='Fixed synapses: perturbation decays');ax.set_ylim(1e-13,.2);ax.set_xlim(0,280);ax.set_title('B. Feedback determines growth versus decay');ax.set_xlabel('Time (ms)');ax.set_ylabel('Absolute perturbation (mV, log scale)');ax.legend(fontsize=7)
ax=axes[1,0]
# Exact independent-cell spectra with all synaptic release fixed.
v=z['v'];m=z['m'];h=z['h'];n=z['n'];N=len(v);am,bm,ah,bh,an,bn=rates(v);dm,dh,dn=[np.imag(x)/1e-20 for x in gates(v.astype(complex)+1e-20j)];C=z['cap'];L=np.zeros((N,4,4));L[:,0,0]=-z['G']/C;L[:,0,1]=-z['gNa']*3*m*m*h*(v-z['eNa'])/C;L[:,0,2]=-z['gNa']*m**3*(v-z['eNa'])/C;L[:,0,3]=-z['gK']*4*n**3*(v-z['eK'])/C
for j,(rate,prime) in enumerate([(am+bm,dm),(ah+bh,dh),(an+bn,dn)]):L[:,j+1,0]=rate*prime;L[:,j+1,j+1]=-rate
ev=np.linalg.eigvals(L);ax.scatter(ev.real.ravel(),ev.imag.ravel(),s=5,color='#2877b6',alpha=.12,label='Fixed synapses: all cells stable')
for k,color,label in [('0.1','#c43a32','Full network, k_minus=0.1'),('0.01','#e28c1c','Full network, k_minus=0.01')]:
 ww=modes['models'][k]['continuous_modes'];x=np.array([a['real_per_ms'] for a in ww]);im=np.array([a['imag_per_ms'] for a in ww]);ax.scatter(np.r_[x,x],np.r_[im,-im],s=28,color=color,label=label)
ax.axvline(0,color='black',lw=1);ax.axvspan(0,.08,color='red',alpha=.06);ax.set_xlim(-.45,.065);ax.set_ylim(-.4,.4);ax.set_title('C. Positive real eigenvalues: oscillatory instability');ax.set_xlabel('Real part of eigenvalue (1/ms)');ax.set_ylabel('Imaginary part (1/ms)');ax.legend(fontsize=7,loc='upper left')
ax=axes[1,1];items=spectrum['models']['0.1']['discrete'];ds=sorted(float(k) for k in items);gg=[max(items[str(d)],key=lambda x:x['modulus'])['growth_per_ms'] for d in ds];ax.semilogx(ds,gg,'o-',color='#c43a32',label='Actual discrete update: log(|mu|) / dt');ax.axhline(growth,color='black',ls='--',label='Continuous-time limit');ax.axhline(0,color='gray',lw=.8);ax.set_ylim(-.002,.066);ax.set_title('D. Smaller dt approaches positive growth, not zero');ax.set_xlabel('Integration step dt (ms)');ax.set_ylabel('Oscillation growth rate (1/ms)');ax.set_xticks(ds,[f'{x:g}' for x in ds]);ax.legend(fontsize=7)
for ax in axes.flat:ax.grid(alpha=.15)
fig.suptitle('Original trained Point HH: recurrent synaptic feedback creates an unstable oscillatory mode\nConstant background; no light switches, no moving bar, no added noise',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'oscillation_proof.png',dpi=155);plt.close(fig)
print(OUT/'oscillation_proof.png')
