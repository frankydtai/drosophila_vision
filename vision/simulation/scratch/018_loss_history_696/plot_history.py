from pathlib import Path
import csv,hashlib,json,os,re
import numpy as np
O=Path(__file__).resolve().parent;SIM=O.parents[1];RUNS=SIM/'0_runs/point_hh';S=SIM/'scratch'
os.environ['MPLCONFIGDIR']=str(O/'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from PIL import Image
segments=[];hashes={}
def track(p):
 p=Path(p);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read_json(p):return json.loads(track(p).read_text())
def read_array(p):return np.load(track(p))
def add(name,phase,before,terminal,source,objective='moving-bar'):
 before=np.asarray(before,dtype=np.float64).reshape(-1)
 assert len(before)>0 and np.isfinite(before).all() and (before>0).all()
 assert np.isfinite(terminal) and terminal>0
 start=sum(len(s['before']) for s in segments)
 segments.append(dict(name=name,phase=phase,objective=objective,start=start,end=start+len(before),before=before,terminal=float(terminal),source=str(source)))
# The four original runs; terminal loss comes from the final evaluation,
# not costs.npy, which stores the selected best checkpoint.
for rid,n,phase,objective in [('34430456',100,'A','spread'),('34430984',50,'B','moving-bar'),('34431163',100,'B','moving-bar'),('34432972',50,'B','moving-bar')]:
 run=next(RUNS.glob(rid+'-*'));before=read_array(run/'data/best_costs.npy');assert len(before)==n
 log=track(SIM/'0_logs'/f'{rid}.log').read_text()
 final=re.findall(r'^Final cost =\s*([0-9.eE+-]+)',log,flags=re.MULTILINE);assert len(final)==1,(rid,final)
 add(rid,phase,before,float(final[0]),run,objective)
# Parallel jobs are serialized in case-ID order, not by wall-clock finish.
for name,n in [('resume',20),('reset',20),('resume_small',20),('reset_small',10)]:
 p=S/'012_adam_resume_diagnosis'/f'{name}.json';d=read_json(p);assert len(d['records'])==n
 add('diagnosis '+name,'C',[r['cost'] for r in d['records']],d['final_cost'],p)
replay=S/'013_save_reset469101';before=read_array(replay/'run/data/best_costs.npy');assert len(before)==20
d=read_json(replay/'replay_result.json');add('replay reset 20','D',before,d['selected_cost'],replay)
for folder,phase in [('014_continue_469101','E'),('015_continue_394150','F')]:
 root=S/folder
 files=['screen_cases.json','refine_cases.json']+(['extend_cases.json'] if phase=='F' else [])
 for case_file in files:
  for case in read_json(root/case_file):
   p=root/case['label']/'progress.json';d=read_json(p);assert d.get('complete'),p
   before=[r['cost'] for r in d['records']]
   # Verify the tabular run history agrees with the per-iteration log.
   stored=read_array(p.parent/'run/data/best_costs.npy');assert np.array_equal(stored,np.asarray(before))
   add(case['label'],phase,before,d['terminal_cost'],p)
assert len(segments)==22,len(segments)
assert sum(len(s['before']) for s in segments)==696
phase_counts={p:sum(len(s['before']) for s in segments if s['phase']==p) for p in 'ABCDEF'}
assert phase_counts==dict(A=100,B=200,C=70,D=20,E=128,F=178),phase_counts
assert sum(len(s['before']) for s in segments if s['objective']=='moving-bar')==596
rows=[];curves={};running={};phase_end={}
for s in segments:
 after=np.r_[s['before'][1:],s['terminal']]
 x=np.arange(s['start'],s['end']+1)
 y=np.r_[s['before'],s['terminal']]
 curves[s['name']]=(x,y)
 objective=s['objective'];prior=running.get(objective,np.inf)
 best=np.minimum.accumulate(np.r_[prior,y])[1:];running[objective]=float(best[-1])
 s['best_x']=x;s['best_y']=best
 phase_end[s['phase']]=s['end']
 for j,(pre,post) in enumerate(zip(s['before'],after)):
  rows.append(dict(iteration=s['start']+j+1,phase=s['phase'],trial=s['name'],trial_iteration=j+1,objective=objective,loss_before=float(pre),loss_after=float(post),source=s['source']))
assert len(rows)==696 and [r['iteration'] for r in rows]==list(range(1,697))
latest=RUNS/'34437673-continued_optimization-loss-282967'
final_saved=float(read_array(latest/'data/costs.npy')[0])
assert segments[-1]['terminal']==final_saved==282966.9375
assert running['moving-bar']==final_saved,(running,final_saved)
with (O/'iterations_696.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(15.8,7.8));fig.subplots_adjust(left=.075,right=.975,bottom=.23,top=.80)
colors={'A':'#f5c788','B':'#c7dae9','C':'#d4cde6','D':'#ebd8cd','E':'#c4e3da','F':'#c5d4eb'}
phase_info={'A':'A  Spread initialization (100)','B':'B  Original moving-bar runs (200)','C':'C  Adam comparisons (70)','D':'D  Exact replay (20)','E':'E  Optimization round 1 (128)','F':'F  Optimization round 2 (178)'}
for phase in 'ABCDEF':
 ss=[s for s in segments if s['phase']==phase];a=ss[0]['start'];b=ss[-1]['end']
 ax.axvspan(a,b,color=colors[phase],alpha=.15,zorder=0)
 ax.axvspan(a,b,ymin=1.025,ymax=1.105,color=colors[phase],alpha=.95,clip_on=False,zorder=3)
 ax.text((a+b)/2,1.065,f'{phase}\n{phase_counts[phase]}',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=10,fontweight='bold',linespacing=1.25)
for s in segments:
 x,y=curves[s['name']]
 raw_color='#c9974c' if s['objective']=='spread' else '#718596'
 ax.plot(x,y,color=raw_color,lw=1.0,alpha=.67,zorder=2)
 ax.scatter([x[0]],[y[0]],color=raw_color,s=9,alpha=.7,zorder=3)
 if s['start']>0:ax.axvline(s['start'],color='#93a0ac',lw=.6,ls=(0,(2,4)),alpha=.5,zorder=1)
# Best-so-far stays separate across the task change at iteration 100.
for objective,color in [('spread','#af6b1f'),('moving-bar','#145dac')]:
 pieces=[s for s in segments if s['objective']==objective]
 x=np.concatenate([s['best_x'] for s in pieces]);y=np.concatenate([s['best_y'] for s in pieces])
 ax.plot(x,y,color=color,lw=2.8,zorder=5)
ax.axvline(100,color='#a97135',lw=1.3,ls='--',zorder=4)
ax.text(104,.94,'Loss objective changes',transform=ax.get_xaxis_transform(),color='#825c2e',fontsize=10,va='top')
ax.scatter([696],[final_saved],s=62,color='#145dac',edgecolor='white',linewidth=1.1,zorder=7)
ax.annotate('Best loss: 282,966.9375',xy=(696,final_saved),xytext=(-158,52),textcoords='offset points',ha='left',va='center',fontsize=11,fontweight='bold',color='#145dac',bbox=dict(boxstyle='round,pad=.4',fc='white',ec='#c9d8e7',alpha=.97),arrowprops=dict(arrowstyle='->',color='#145dac',lw=1.2),zorder=8)
ax.set_yscale('log');ymax=max(float(np.max(s['before'])) for s in segments)*1.13
ax.set_ylim(2.4e5,ymax);ax.set_xlim(0,708)
ax.set_xticks(list(range(0,701,50)));ax.set_xlabel('Cumulative optimization iterations (all trials included)',labelpad=12)
ax.set_ylabel('Loss (log scale)',labelpad=10)
ticks=[3e5,5e5,1e6,2e6,5e6,1e7,2e7,5e7];ax.yaxis.set_major_locator(FixedLocator([v for v in ticks if 2.4e5<v<ymax]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v/1e6:g}M'));ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(axis='y',which='major',alpha=.2,lw=.7);ax.grid(axis='x',alpha=.1,lw=.6)
handles=[Line2D([0],[0],color='#718596',lw=1.2,label='Measured loss within each trial'),Line2D([0],[0],color='#af6b1f',lw=2.8,label='Best seen: spread'),Line2D([0],[0],color='#145dac',lw=2.8,label='Best seen: moving bar')]
ax.legend(handles=handles,loc='upper right',framealpha=.97,fontsize=10)
fig.suptitle('Loss history across 696 optimization iterations',y=.975,fontsize=19,fontweight='bold')
fig.text(.5,.918,'Stage order; parallel trials arranged by case ID. Trial boundaries are dotted; curves do not connect across restarts.',ha='center',fontsize=10.5,color='#4b5563')
phase_handles=[Patch(facecolor=colors[p],edgecolor='none',label=phase_info[p]) for p in 'ABCDEF']
fig.legend(handles=phase_handles,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=10.5,columnspacing=2.1,handlelength=1.4)
fig.text(.075,.035,'The first 100 iterations use the spread objective; the next 596 use moving-bar loss. Best-so-far resets at the objective switch.',fontsize=9.5,color='#4b5563')
fig.text(.075,.012,'Includes comparison branches and the 20-iteration replay; excludes extra loss evaluations and backtracking probes. No smoothing.',fontsize=9.5,color='#4b5563')
for ext in ['png','pdf']:
 fig.savefig(O/f'loss_history_696.{ext}',dpi=190,facecolor='white')
plt.close(fig)
with Image.open(O/'loss_history_696.png') as im:im.verify()
with Image.open(O/'loss_history_696.png') as im:size=[im.width,im.height]
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
metadata=dict(total_iterations=696,phase_counts=phase_counts,trial_count=len(segments),moving_bar_iterations=596,spread_iterations=100,final_best_loss=final_saved,parallel_order='case-ID order within each phase',samples='Loss before and after each optimization iteration; endpoint evaluations do not add iterations.',objectives_comparable_across_switch=False,source_files_unchanged=True,image_size=size,segments=[{k:v for k,v in s.items() if k not in ['before','best_x','best_y']} for s in segments])
(O/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(O/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({k:v for k,v in metadata.items() if k!='segments'},indent=2))
