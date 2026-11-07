from pathlib import Path
import json,ast
O=Path(__file__).resolve().parent
s=(O/'worker.py').read_text()
anchor="started=time.monotonic();records=[]"
extra="""baseline_response=json.loads((O/'source_metrics.json').read_text())['groups']
peak_baseline={}
for key in source_parts:
 cell,label=key.split('_')[-2:]
 peak_baseline[key]=baseline_response['mbar_DSI_bright_'+cell][label.lower()+'_peak_mV']
current_peaks={};best_peaks={};best_peak_score=float('inf')
original_forward=costmod.forward_pack
def forward_with_peaks(session,params,i_sti,pack):
 trace=original_forward(session,params,i_sti,pack)
 if pack.task=='mbar':
  _,bias=costmod.gt_affine_from_pack(params,pack,session)
  stops=pack.dsi_t0s+pack.dsi_n_ts;before=(pack.dsi_n_ts-1)//4;first=pack.dsi_t0s+before
  offsets=torch.arange(-int(before.max()),int((pack.dsi_n_ts-1-before).max())+1,device=trace.device)
  ts=first[:,None]+offsets[None,:];valid=(ts>=pack.dsi_t0s[:,None])&(ts<stops[:,None])
  response=trace[pack.entry_bs[:,None],ts.clamp(0,trace.shape[1]-1),pack.entry_nodes[:,None]]-bias[:,None]
  for key in dict.fromkeys(pack.entry_part_keys):
   mask=torch.tensor([k==key for k in pack.entry_part_keys],device=trace.device)&(pack.cost_scales>0)
   if bool(mask.any()):current_peaks[key]=response[mask].mean(0)[valid[mask].all(0)].amax().clamp_min(0)
 return trace
costmod.forward_pack=forward_with_peaks
"""
s=s.replace(anchor,extra+anchor)
s=s.replace('global best_cost,best_z,best_state,best_parts,best_at,best_nd,unconstrained_best','global best_cost,best_z,best_state,best_parts,best_at,best_nd,unconstrained_best,best_peaks,best_peak_score')
s=s.replace(" if eligible and (nd,value)<(best_nd,best_cost):\n  best_nd=nd", """ peaks={k:float(v.detach()) for k,v in current_peaks.items()}
 peak_score=sum(peaks[k]/peak_baseline[k] for k in nd_keys)
 peak_ok=all(peaks[k]<peak_baseline[k] for k in nd_keys)
 ratio_ok=all(peaks[k]/peaks[k[:-2]+'PD']<peak_baseline[k]/peak_baseline[k[:-2]+'PD'] for k in nd_keys)
 if eligible and (at==0 and best_z is None or peak_ok and ratio_ok and peak_score<best_peak_score):
  best_peak_score=peak_score;best_peaks=peaks;best_nd=nd""")
s=s.replace('report.update(best_cost=best_cost,','report.update(best_peak_score=best_peak_score,best_peaks=best_peaks,best_cost=best_cost,')
s=s.replace("c=float(value.detach());floats={k:float(v.detach()) for k,v in parts.items()};value.backward();del value,parts", """c=float(value.detach());floats={k:float(v.detach()) for k,v in parts.items()}
  objective=value/source_total+case['nd_peak_weight']*sum((current_peaks[k]/peak_baseline[k])**2 for k in nd_keys)+case['pd_peak_weight']*sum((current_peaks[k]/peak_baseline[k]-1)**2 for k in source_parts if k.endswith('_PD'))
  objective.backward();del objective,value,parts""")
s=s.replace("selection='lowest_ND_sum_with_both_ND_parts_and_total_at_most_source'", "selection='lowest_normalized_ND_peaks_with_original_total_ND_losses_and_ND_PD_ratios_improved',training_objective='original_MSE/source_MSE + ND_peak_weight*sum((ND/source_ND)^2) + PD_peak_weight*sum((PD/source_PD-1)^2)',optimizer_state_objective='temporary_peak_objective_reset_Adam_when_returning_to_plain_MSE',selected_peaks=best_peaks")
ast.parse(s);(O/'peak_worker.py').write_text(s)
source=json.loads((O/'campaign_state.json').read_text())['source']
cases=[dict(label='nd_peak_constrained',source=source,native=False,resume_adam=False,lr=.0001,n_iter=24,backtrack=False,nd_peak_weight=.5,pd_peak_weight=20.,tasks=['mbar'],contrasts=['bright'],part_cost_scales={'mbar':1.,'mbar_DSI':0.})]
(O/'peak_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
s=(O/'screen.slurm').read_text().replace('--array=0-2','--array=0').replace('hh_init_nd','hh_nd_peak').replace('--mem=32G','--mem=64G').replace('screen-%A_%a.log','peak-%A_%a.log').replace('worker.py screen_cases.json','peak_worker.py peak_cases.json');(O/'peak.slurm').write_text(s)
p=O/'choose_result.py';s=p.read_text().replace("['screen_cases.json','refine_cases.json']","['screen_cases.json','refine_cases.json','peak_cases.json']");p.write_text(s)
print('Temporary peak objective prepared; original reported loss weights preserved.')
