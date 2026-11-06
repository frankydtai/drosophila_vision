from pathlib import Path
import json,statistics,hashlib
O=Path(__file__).resolve().parent
rows=[]
for p in sorted(O.glob('case_*.json')):
 a=json.loads(p.read_text());rr=a['iterations'];costs=[r['cost'] for r in rr if r.get('cost') is not None]
 if not costs:continue
 initial=costs[0];points=[]
 for r in rr:
  elapsed=r.get('train_elapsed_s')
  if elapsed is None:elapsed=sum(x['iteration_s'] for x in rr[:r['iteration']+1])
  if r.get('cost') is not None:points.append((elapsed,r['cost']))
  if r.get('accepted_cost') is not None:points.append((elapsed,r['accepted_cost']))
 best=min(v for t,v in points)
 samples=rr[2:] if len(rr)>3 else rr
 row=dict(file=p.name,run=a['case']['run'],lr=a['case']['lr'],device=a['device'],label=a['label'],n=len(rr),limit=a['limit'],finished=len(rr)==a['limit'] or a['stop_reason']!='completed',stop=a['stop_reason'],initial=initial,best=best,reduction_percent=100*(1-best/initial),final_cost=points[-1][1],median_iteration_s=statistics.median(r['iteration_s'] for r in samples),median_forward_s=statistics.median(r['forward_s'] for r in samples if r.get('forward_s') is not None) if any(r.get('forward_s') is not None for r in samples) else None,time_to_20pct_s=next((t for t,v in points if v<=initial*.8),None),time_to_40pct_s=next((t for t,v in points if v<=initial*.6),None),time_to_60pct_s=next((t for t,v in points if v<=initial*.4),None),source_unchanged=a['source_files_unchanged'],excluded=a.get('excluded_from_ranking'))
 rows.append(row)
hashes=json.loads((O/'source_hashes.json').read_text());changed=[p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
result=dict(integrator_step_ms=1.0,pre_step_ms=1.0,fp=32,formal_files_changed_since_snapshot=changed,rows=rows)
(O/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for a in rows:
 if 'extended' in a['label'] or 'backtrack' in a['label']:
  print(a['file'],a['n'],a['stop'],round(a['reduction_percent'],2),'iter',round(a['median_iteration_s'],2),'fwd',round(a['median_forward_s'] or 0,2),'t20',a['time_to_20pct_s'],'t40',a['time_to_40pct_s'])
print('FORMAL_FILES_CHANGED',changed)
