"""Use the same joint objective as the GPU run on CPU (no source edits)."""
from pathlib import Path
from dataclasses import replace
import sys,runpy
SIM=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SIM))
import import_bootstrap
import torch
import train.implementation as implementation
original=implementation.build_session

def joint_session(*args,**kwargs):
    session=original(*args,**kwargs)
    if str(session.device).startswith('cpu'):
        session=replace(session,sequential=False,train_opts=dict(session.train_opts,sequential=False))
    return session

implementation.build_session=joint_session
torch.set_num_threads(1)
runpy.run_path(str(SIM/'run_init_from.py'),run_name='__main__')
