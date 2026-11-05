from pathlib import Path
import runpy,json
ROOT=Path(__file__).resolve().parent
ns=runpy.run_path(str(ROOT/'analyze.py'))
ns['torch'].set_num_threads(1)
SIM=ns['SIM'];plot=ns['plot'];stats=ns['stats']
curves={}
for label,run in [('before',Path((ROOT/'baseline_run.txt').read_text().strip())),('after',SIM/'0_runs/hp_lp/34390303-ab34381383-cd-transfer-keepab-iter-300')]:
    session,_,_=plot.load_best(str(ns['SOURCE']))
    values,cells,pairs=ns['load_best_node_vals'](str(run))
    session,z=ns['session_z_from_node_vals'](values,cells,pairs,session)
    info,curves[label]=stats(session,z)
plt=ns['plt'];fig,axes=plt.subplots(2,2,figsize=(9,5),sharex=True,sharey='row')
for i,c in enumerate(['T4a','T4b']):
    for j,d in enumerate(['PD','ND']):
        t,p,g=curves['after'][c,d];_,old,_=curves['before'][c,d];ax=axes[i,j]
        ax.plot(t,g,color='black',label='GT');ax.plot(t,old,color='#3569b0',label='Before');ax.plot(t,p,color='#da7426',label='After ABCD 300')
        ax.set_title(c+' '+d);ax.set_xlim(-150,576);ax.grid(alpha=.15)
        if j==0:ax.set_ylabel('Vm - cell onset bias (mV)')
        if i==1:ax.set_xlabel('Time from local stimulus onset (ms)')
axes[0,0].legend(fontsize=8,frameon=False)
fig.suptitle('Original AB evaluation: same original horizontal stimulus and entries',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ROOT/'AB_original_test_before_after.png',dpi=150)
