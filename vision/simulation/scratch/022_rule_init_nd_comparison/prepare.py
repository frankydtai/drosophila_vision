from pathlib import Path
import shutil,json,hashlib,subprocess,ast
O=Path(__file__).resolve().parent;P=O.parent/'021_continue_init_nd';R=O.parents[3]
shutil.copytree(P/'copy',O/'copy',symlinks=True)
protected={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in json.loads((P/'protected_hashes.json').read_text())}
(O/'protected_hashes.json').write_text(json.dumps(protected,indent=2)+'\n');(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
s=(P/'response_metrics.py').read_text();s=s[:s.index("if __name__=='__main__':")]
s=s.replace('def evaluate(run, snapshot=None):','def evaluate(run, snapshot=None, override=None):')
anchor=' with torch.no_grad():\n  parts='
change=""" if override:
  import copy
  from dataclasses import replace
  import neuron.point_hh as hh
  original=session.model_context;other_opts=copy.deepcopy(opts)
  original_mode=original.provenance['g_context_mode'];other_opts['point_hh']['g_context_mode']='rule' if original_mode=='init' else 'init'
  other=hh.build_context(session.connectome,other_opts)
  params=dict(original.params)
  if override in ['membrane','all']:
   for name in ['gNa','gK','gLeak']:params[name]=other.params[name].clone()
  if override in ['synapse','all']:params['gS']=other.params['gS'].clone()
  if override=='close_unresolved':params['gS']=original.params['gS']*(other.params['gS']>0)
  if override=='restore_Mi4_Mi9_T4':
   mask=torch.tensor([row['pre_type'] in ['Mi4','Mi9'] and row['post_type'].startswith('T4') and row['e_syn_mV']<-40 for row in original.edge_rows],device=session.device)
   params['gS']=torch.where(mask,other.params['gS'],original.params['gS'])
  session=replace(session,model_context=replace(original,params=params))
 with torch.no_grad():
  parts="""
assert s.count(anchor)==1;s=s.replace(anchor,change);ast.parse(s);(O/'response_metrics.py').write_text(s)
root=R/'vision/simulation/0_runs/point_hh';rule=str(root/'34437673-continued_optimization-loss-282967');init=str(root/'34466728-init_nd_optimization-loss-994279')
cases=[dict(label='rule_best',run=rule,override=None),dict(label='init_best',run=init,override=None)]
for override in ['synapse','membrane','all','close_unresolved','restore_Mi4_Mi9_T4']:cases.append(dict(label='init_weights_'+override,run=init,override=override))
cases.append(dict(label='rule_weights_init_baseline',run=rule,override='all'))
(O/'cases.json').write_text(json.dumps(cases,indent=2)+'\n');print('Prepared eight no-training, fixed-weight comparisons.')
