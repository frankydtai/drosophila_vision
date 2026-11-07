from pathlib import Path
import json,shutil,ast
O=Path(__file__).resolve().parent;P=O.parent/'021_continue_init_nd';Q=O.parent/'022_rule_init_nd_comparison';G=O.parent/'024_mi4_gs_upper_bound'
# All evaluation code is local to this campaign and uses the updated snapshot.
for name in ['response_metrics.py','parameter_report.py']:shutil.copy2(Q/name,O/name)
shutil.copy2(G/'upper_only_6/metrics.json',O/'old_mapping_metrics.json');shutil.copy2(G/'upper_only_6/curves.npz',O/'old_mapping_curves.npz');shutil.copy2(G/'configured_10/metrics.json',O/'new_mapping_metrics.json');shutil.copy2(G/'configured_10/curves.npz',O/'new_mapping_curves.npz')
s=(P/'render.py').read_text();s=s.replace("result=dict(selected_run=", "assert opts['point_hh']['gene']['max_fold_changes']['gS']==10.0\nassert session.model_context.provenance['gene_mapping']['max_fold_changes']['gS']==10.0\nresult=dict(authorized_config_change='gS fold 6 to 10',selected_run=");(O/'render.py').write_text(s)
s=(P/'finalize.slurm').read_text().replace('021_continue_init_nd','025_retrain_init_gs_fold10').replace('hh_nd_finalize','hh_gs10_finalize');(O/'finalize.slurm').write_text(s)
s=(P/'assess.slurm').read_text().replace('021_continue_init_nd','025_retrain_init_gs_fold10').replace('hh_nd_assess','hh_gs10_assess');(O/'assess.slurm').write_text(s)
for name in ['response_metrics.py','render.py']:ast.parse((O/name).read_text())
p=O/'campaign_state.json';d=json.loads(p.read_text());d['screen_job']='34467927';p.write_text(json.dumps(d,indent=2)+'\n')
print('Prepared model reload, Adam verification, and full output plotting workflow.')
