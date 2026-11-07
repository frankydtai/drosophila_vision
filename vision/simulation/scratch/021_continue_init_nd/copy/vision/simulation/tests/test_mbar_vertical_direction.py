"""Anatomical direction checks independent of the bar implementation.

FAFB column p+1,q+1 goes dorsally; build_hex maps this to y-1 in
both hemispheres. An upward stimulus must reach that hex first.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
import import_bootstrap  # noqa: E402,F401
from build_hex import uv_from_pq, xy_from_uv  # noqa: E402
from task.mbar.gt import motion_preference, fig1_trace_from_sti  # noqa: E402
from task.mbar.sti_geo import bar_bound0_bar_bound1s  # noqa: E402
from task.mbar.sti_spec import (  # noqa: E402
    MbarSpec, build_bar_axis_distance_hex, build_i_sti_hex,
    hex_first_sti_t, mbar_n_t, mbar_transit_times, _mbar_cache_digest,
)


@pytest.mark.parametrize('side', ['right', 'left'])
def test_fafb_dorsal_neighbor_is_negative_y(side):
    u,v = uv_from_pq(np.array([0,1]), np.array([0,1]), side)
    x,y = xy_from_uv(u,v)
    np.testing.assert_allclose(x, [0,0])
    np.testing.assert_allclose(y, [0,-1])


@pytest.mark.parametrize('direction,first,last', [('up',2,0), ('down',0,2)])
@pytest.mark.parametrize('peak', [40., 0.])
def test_vertical_stimulus_reaches_anatomical_origin_first(direction, first, last, peak):
    # Index 0 is dorsal, index 2 ventral. Upward travels ventral -> dorsal.
    hexes = [SimpleNamespace(x=0.,y=y) for y in [-1.,0.,1.]]
    specs = [MbarSpec(direction, 'bright' if peak else 'dark', 2.25)]
    kwargs = dict(bar_dist=4, multi_bar=False, t_onset=162, delta_ms=2.,
                  view_deg=(-6.,-8.,6.,8.))
    n_t = mbar_n_t(specs, kwargs['view_deg'], 4, multi_bar=False,
                  t_onset=162, delta_ms=2.)
    stimulus = build_i_sti_hex(hexes, specs, n_t, i_baseline=20., i_sti=peak, **kwargs)
    onsets = [hex_first_sti_t(stimulus[0,:,i],20.) for i in range(3)]
    assert onsets[first] < onsets[1] < onsets[last]
    distances = build_bar_axis_distance_hex(hexes, specs, n_t, **kwargs)
    peaks = distances[0].argmin(axis=0)
    assert peaks[first] < peaks[1] < peaks[last]


@pytest.mark.parametrize('direction,first,last', [('right',0,2), ('left',2,0)])
def test_horizontal_stimulus_keeps_original_orientation(direction,first,last):
    hexes = [SimpleNamespace(x=x,y=0.) for x in [-1.,0.,1.]]
    specs = [MbarSpec(direction,'bright',2.25)]
    kwargs = dict(bar_dist=4,multi_bar=False,t_onset=162,delta_ms=2.,view_deg=(-8.,-6.,8.,6.))
    stimulus = build_i_sti_hex(hexes,specs,700,i_baseline=20.,i_sti=40.,**kwargs)
    onsets = [hex_first_sti_t(stimulus[0,:,i],20.) for i in range(3)]
    assert onsets[first] < onsets[1] < onsets[last]


@pytest.mark.parametrize('direction,expected_start', [('up',1),('down',-1)])
def test_multibar_sweeps_start_at_correct_vertical_boundary(direction,expected_start):
    bounds=bar_bound0_bar_bound1s(MbarSpec(direction,'bright',2.25),(-20.,-20.,20.,20.),bar_dist=1,multi_bar=True)
    assert np.sign(np.mean(bounds[0])) == expected_start


@pytest.mark.parametrize('side', ['right','left'])
def test_vertical_gt_keeps_c_up_and_d_down(side):
    for cell,pd,nd in [('T4c','up','down'),('T4d','down','up')]:
        assert motion_preference(side,cell,pd,'bright').pd_nd == 'PD'
        assert motion_preference(side,cell,nd,'bright').pd_nd == 'ND'
        assert fig1_trace_from_sti(side,cell,pd,'bright',2.25) == 'T4_PC_w1_PD'
        assert motion_preference(side,cell,'right','bright') is None


def test_cache_records_anatomical_direction_version(tmp_path):
    import hashlib,json
    source=tmp_path/'network.json';source.write_text('{}')
    specs=[MbarSpec('up','bright',2.25)]
    stat=source.stat()
    legacy=dict(network=str(source.resolve()),network_mtime_ns=stat.st_mtime_ns,network_size=stat.st_size,
                hex_uv=[(0,0)],bar_dist=4,multi_bar=False,view_deg=None,
                specs=[dict(direction='up',contrast='bright',bar_w_deg=2.25,speed_deg_over_s=56.25)],
                n_t=700,t_onset=162,delta_ms=2.,i_baseline=20.,i_sti=40.)
    old=hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()[:16]
    new=_mbar_cache_digest(source,specs,[(0,0)],700,162,2.,20.,4,False,40.,None)
    assert new != old


def test_corrected_geometry_resets_automatic_adam_resume():
    from train.implementation import objective_diff, resolve_resume_adam
    old={'tasks':['mbar'],'mbar_sti_opts':{'bar_directions':['up','down']}}
    new={'tasks':['mbar'],'mbar_sti_opts':{'bar_directions':['up','down'],
                                       'geometry_version':'fafb_y_ventral_v2'}}
    assert objective_diff(old,new) == ['mbar_geometry_version']
    use_adam,_ = resolve_resume_adam('auto',same_objective=not objective_diff(old,new))
    assert not use_adam


def test_c_d_real_pack_assigns_expected_gt_and_zero_loss(tmp_path):
    import torch
    from task.mbar.pack import build_mbar_gt, MbarPack
    from train.session import materialize_pack
    from train.cost import _pack_cost_parts_from_v_readout
    from neuron.readout import pack_traces
    # Three columns on the dorsal/ventral axis, each with receptor, T4c, T4d.
    us=np.repeat([-1,0,1],3);vs=np.zeros(9,dtype=int);ids=np.tile(np.arange(3),3)
    names=['R1-6','T4c','T4d']
    network=SimpleNamespace(cells=names,us=us,vs=vs,node_cells=torch.tensor(ids),n_node=9,
                            sti_nodes=np.array([0,3,6]),meta={'side':'right'},source_json=None,
                            sti_nodes_at_uv=lambda u,v:np.flatnonzero((us==u)&(vs==v)&(ids==0)),
                            nodes_at_uv=lambda u,v,cell,**kw:np.flatnonzero((us==u)&(vs==v)&(ids==names.index(cell))))
    fig=tmp_path/'gt.npz'
    np.savez(fig,**{f'T4_PC_w1_{d}__{field}':np.array(values) for d,peak in [('PD',4.),('ND',1.)]
                   for field,values in [('time_ms',[0.,324.,400.,900.]),('vm_mv',[0.,0.,peak,0.])]})
    gt=build_mbar_gt(network,t_onset=162,delta_ms=2.,use_cache=False,bar_dist=4,
                     bar_ws_deg=[2.25],bar_directions=['up','down'],multi_bar=False,
                     i_baseline=20.,i_sti=40.,contrasts=['bright'],gt_cells=['T4c','T4d'],fig1_path=fig)
    for b,node,trace,label in zip(gt.entry_bs,gt.entry_nodes,gt.gts,gt.cost_pd_nds):
        expected_pd=(b==0 and names[ids[node]]=='T4c') or (b==1 and names[ids[node]]=='T4d')
        assert label == (0 if expected_pd else 1)
        assert trace.max() == (4. if expected_pd else 1.)
    pack=materialize_pack(MbarPack(task='mbar',contrast='bright',i_sti=gt.i_sti,gts=gt.gts,
        cost_scales=gt.cost_scales,entry_bs=gt.entry_bs,entry_nodes=gt.entry_nodes,
        cost_t0s=gt.cost_t0s,cost_pd_nds=gt.cost_pd_nds,entry_part_keys=gt.entry_part_keys,
        t_onset=gt.t_onset),device='cpu',sim_dtype=torch.float64)
    perfect=torch.zeros_like(pack.i_sti)
    for b,node,t0,target in zip(pack.entry_bs,pack.entry_nodes,pack.cost_t0s,pack.gts):
        perfect[b,t0:t0+target.numel(),node]=target
    perfect.requires_grad_()
    session=SimpleNamespace(tasks=['mbar'],part_cost_scales={'mbar':1.,'mbar_DSI':0.},
                            train_opts={'cost_norm':'a_gt2','cost_entry_reduce':'mean_trace'},sim_dtype=torch.float64)
    parts=_pack_cost_parts_from_v_readout(pack,session,torch.ones_like(pack.cost_scales),
                                         torch.zeros_like(pack.cost_scales),pack_traces(perfect,pack),trace=perfect)
    loss=sum(parts.values());loss.backward()
    assert float(loss)==0.
    assert torch.isfinite(perfect.grad).all()
