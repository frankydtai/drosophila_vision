"""Follow up the observed k=0.01 benefit, keeping the same exact HH model."""
from pathlib import Path
import json,gc
import interventions as ex
import validation as val
OUT=Path(__file__).resolve().parent
rows=[]
def save(r):
 rows.append(r);(OUT/'k001_extension_results.json').write_text(json.dumps(rows,indent=2)+'\n');print('DONE',r['case']['name'],r['cost'],r['dsi_raw'],r['peak_metrics'],flush=True)
if __name__=='__main__':
 for dose in [.25,.5,1.5,2.]:
  case=dict(name=f'k001_negative_{dose:g}',k_label='HH_k001',negative_sources=['Mi1','Tm3'],dose=dose);v,p,g=ex.conditional(case);save(ex.evaluate(case,v,p,g));del v,p,g;gc.collect()
 case=dict(name='full_k001_negative_1',k_label='HH_k001',negative_sources=['Mi1','Tm3'],dose=1.,check_conditional=True);save(val.full(case))
 print('K001_EXTENSION_FINISHED',flush=True)
