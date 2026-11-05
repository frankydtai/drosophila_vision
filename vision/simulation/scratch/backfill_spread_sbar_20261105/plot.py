from pathlib import Path
import hashlib
import json
import sys
SIM=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SIM))
import import_bootstrap
import torch
import figure.plot as plot

torch.set_num_threads(1)
runs=[
    '34382310-t4abcd-from-34381383-iter-0',
    '34382311-t4abcd-from-34381383-iter-300',
    '34382547-t4abcd-direct-from-34376877-iter-0',
    '34382547-t4abcd-direct-from-34376877-iter-300',
]
verified=[]
for token in runs:
    run=SIM/'0_runs/hp_lp'/token
    watched=sorted((run/'data').glob('*'))+[run/'param.csv',run/'syn_strength_cell.csv',run/'mbar_gt_v.png',run/'mbar_all_v.png']
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched if p.is_file()}
    print('Backfilling',run,flush=True)
    plot.plot_trained_run(str(run),figure_kwargs=dict(figure_tasks=['spread','sbar'],plot_right_only=False))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()), 'Existing training data changed'
    for name in ('spread_gt_v.png','spread_all_v.png','sbar_gt_v.png','sbar_all_v.png'):
        p=run/name
        assert p.is_file() and p.stat().st_size>0,p
        verified.append(dict(path=str(p),bytes=p.stat().st_size))
(Path(__file__).resolve().parent/'verified_outputs.json').write_text(json.dumps(verified,indent=2)+'\n')
print('Verified',len(verified),'new figures; existing training data and mbar plots unchanged.',flush=True)
