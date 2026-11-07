from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import importlib.util
import json
import sys

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'vision'))
import import_bootstrap
import train
import figure.mbar as mbar
from task.mbar.sti_spec import MbarSpec

spec=importlib.util.spec_from_file_location('gt_dsi_fixtures', ROOT/'vision/simulation/tests/test_mbar_dsi_cost.py')
fixtures=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

@pytest.mark.parametrize('weight',[0.,1.])
@pytest.mark.parametrize('gain',[0.,1.,-1.])
@pytest.mark.parametrize('paired',[False,True])
def test_gt_every_panel_shows_raw_dsi_independent_of_weight(monkeypatch, weight, gain, paired):
    cells=['T4a','T4b']
    traces={}
    readouts={}
    monkeypatch.setattr(train,'params_from_z',lambda *_: {})
    monkeypatch.setattr(train,'gt_affine_from_cell',lambda *_a,**_kw: (1.,0.))
    monkeypatch.setattr(mbar,'v_th_from_z',lambda *_: {})
    monkeypatch.setattr(mbar,'e_leak_from_z',lambda *_: {})
    monkeypatch.setattr(mbar,'cost_sti_hexes',lambda *_a,**_kw: [])
    monkeypatch.setattr(mbar,'nodes_from_hexes',lambda *_: np.array([0]))
    monkeypatch.setattr(mbar,'_load_mbar_gt_traces',lambda *_: {})
    monkeypatch.setattr(mbar,'_mbar_readout_hexes_label',lambda *_: 'test traces')
    monkeypatch.setattr(train,'forward_pack',lambda _s,_p,_i,p: traces[p.contrast])
    def from_forward(session,task,contrast,trace,specs,tokens,**kwargs):
        means={(cell,token): trace[b,:,c] for c,cell in enumerate(cells) for b,token in enumerate(tokens)}
        wt=mbar.MbarWindowTraces(
            ca_mean_cell=means,ca_sd={k:np.zeros_like(v) for k,v in means.items()},
            ca_n={k:1 for k in means},before_t={token:3 for token in tokens},
            after_t={token:8 for token in tokens},t0_bn=None,cell_idxs=np.array([0,1]),cells=cells)
        return wt,cells,'right',1,0,True
    monkeypatch.setattr(mbar,'_traces_from_forward',from_forward)
    monkeypatch.setattr(mbar,'bar_specs_from_task',lambda _s,_t,c: [MbarSpec(d,c,2.25) for d in ('right','left')])
    monkeypatch.setattr(mbar,'mbar_specs_by_cell',lambda _s,_t,c,_side: {cell:[f'{d}_{c}_w1' for d in ('right','left')] for cell in cells})
    for contrast in (('bright','dark') if paired else ('bright',)):
        pack=replace(fixtures.make_pack(),contrast=contrast,
                     entry_dsi_part_keys=tuple(k.replace('bright',contrast) for k in fixtures.make_pack().entry_dsi_part_keys))
        session=SimpleNamespace(
            packs={'mbar':{contrast:pack}},connectome=SimpleNamespace(cells=cells),
            n_t=12,delta_ms=1.,part_cost_scales={'mbar':1.,'mbar_DSI':weight},
            train_opts={'part_cost_scales':{'mbar':1.,'mbar_DSI':weight}})
        traces[contrast]=fixtures.response_trace()*gain*(2 if contrast=='dark' else 1)
        readout=mbar.mbar_trace_readout(session,torch.zeros(1),'mbar',contrast)
        scale=gain*(2 if contrast=='dark' else 1)
        assert readout.dsi_by_cell == {'T4a':3*scale,'T4b':4*scale}
        readouts[contrast]=readout
    captured=[]
    save=mbar.save_figure
    def capture(fig,path,**kwargs):
        titles=[ax.get_title() for ax in fig.axes if ax.axison]
        expected=[]
        for cell in cells:
            for readout in readouts.values():
                expected.extend([f'DSI={readout.dsi_by_cell[cell]:.3f}']*2)
        assert len(titles)==len(expected)
        assert all(t.endswith(d) for t,d in zip(titles,expected))
        assert all(t.count('DSI=')==1 for t in titles)
        captured.extend(titles)
        save(fig,path,**kwargs)
    monkeypatch.setattr(mbar,'save_figure',capture)
    name=f'gt_weight_{weight:g}_gain_{gain:g}_paired_{paired}'
    output=HERE/f'{name}.png'
    mbar.plot_gt(output,readouts=readouts,title=f'GT: DSI cost weight={weight:g}',cost_parts={})
    assert output.is_file()
    (HERE/f'{name}.json').write_text(json.dumps({'weight':weight,'gain':gain,'paired':paired,'titles':captured},indent=2))
