from pathlib import Path
import shutil,json,hashlib,subprocess
R=Path('/home/yitai/drosophila_vision'); O=R/'vision/simulation/scratch/lr_dt_stability_20261106'; C=O/'copy'
manifest={}
for part in ['vision/simulation','vision/connectome/FAFBv783']:
 for p in (R/part).rglob('*'):
  rel=p.relative_to(R)
  if any(x in rel.parts for x in ['scratch','0_logs','0_runs','__pycache__','.pytest_cache']):continue
  if p.is_file() and p.suffix in ['.py','.yaml','.yml']:
   q=C/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
   manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
p=R/'vision/import_bootstrap.py';q=C/'vision/import_bootstrap.py';shutil.copy2(p,q);manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(C/'figure_digitization').symlink_to(R/'figure_digitization',target_is_directory=True)
network=R/'vision/connectome/FAFBv783/4_built_networks/right_min_neuron1_r10_filtered/network.json'
(O/'inputs').mkdir(exist_ok=True);shutil.copy2(network,O/'inputs/network.json')
for rid in ['34401153','34424399']:
 run=next((R/'vision/simulation/0_runs/point_hh').glob(rid+'-*'))
 dst=O/'inputs'/rid;dst.mkdir(exist_ok=True)
 for name in ['train_opts.json','best_param.npz']:shutil.copy2(run/'data'/name,dst/name)
 opts=json.loads((dst/'train_opts.json').read_text());opts['network_json']=str(O/'inputs/network.json')
 (dst/'isolated_opts.json').write_text(json.dumps(opts,indent=2)+'\n')
(O/'source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
cases=[]
for rid in ['34424399','34401153']:
 for lr,dt in [(.1,1.),(.05,1.),(.03,1.),(.02,1.),(.01,1.),(.003,1.),(.001,1.)]:
  cases.append(dict(run=rid,lr=lr,dt=dt,iterations=20))
(O/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('snapshot',len(manifest),'files;',len(cases),'cases')
