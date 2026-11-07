from pathlib import Path
import shutil,json,hashlib,subprocess
O=Path(__file__).resolve().parent;R=O.parents[3];C=O/'copy'
manifest={}
for part in ['vision/simulation','vision/connectome/FAFBv783']:
 for p in (R/part).rglob('*'):
  rel=p.relative_to(R)
  if any(x in rel.parts for x in ['scratch','0_logs','0_runs','__pycache__','.pytest_cache']):continue
  if p.is_file() and p.suffix in ['.py','.yaml','.yml']:
   q=C/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
p=R/'vision/import_bootstrap.py';shutil.copy2(p,C/'vision/import_bootstrap.py');manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(C/'figure_digitization').symlink_to(R/'figure_digitization',target_is_directory=True)
(O/'protected_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
(O/'git_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=R,text=True))
cases=[dict(label='spread_native_100',native=True,seed=0,source=None,resume_adam=False,lr=.003,n_iter=100,backtrack=False,tasks=['spread'],contrasts=['bright','dark'],part_cost_scales={'mbar':1.,'mbar_DSI':0.})]
(O/'spread_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('Current init-mode code/config snapshotted:',len(manifest),'files.')
