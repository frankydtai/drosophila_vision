from dataclasses import replace
from types import SimpleNamespace
import torch
import pytest
from conftest import pack, fixtures, optimization, HERE


def test_known_peak_ratio_and_existing_raw_loss():
    p=fixtures.make_pack()
    v=fixtures.response_trace().requires_grad_()
    peak=pack.dsi_peak_values(p,v,torch.zeros_like(p.cost_scales))
    assert float(peak[fixtures.KEY_A].detach())==pytest.approx(.6)
    assert float(peak[fixtures.KEY_B].detach())==pytest.approx(.5)
    raw=pack.dsi_values(p,v)
    loss=pack.dsi_cost_parts(p,v)
    assert float(raw[fixtures.KEY_A].detach())==3
    assert float(raw[fixtures.KEY_B].detach())==4
    assert float(loss[fixtures.KEY_A].detach())==-3
    assert float(loss[fixtures.KEY_B].detach())==-4
    sum(peak.values()).backward()
    assert torch.isfinite(v.grad).all()


def test_restores_historical_algorithm_with_current_window_geometry():
    namespace={'torch':torch,'dsi_groups':pack.dsi_groups}
    source=(HERE/'historical_peak_loss.py').read_text().replace('// 3','// 4')
    exec(compile(source,'historical peak loss','exec'),namespace)
    p=fixtures.make_pack()
    v=fixtures.response_trace()+20
    bias=torch.full_like(p.cost_scales,20)
    old=namespace['dsi_cost_parts'](p,v,bias)
    new=pack.dsi_peak_values(p,v,bias)
    for key in old:
        torch.testing.assert_close(old[key],(new[key]-1).square(),rtol=0,atol=0)


def test_baseline_and_positive_gain_invariance():
    p=fixtures.make_pack()
    v=fixtures.response_trace()
    baseline=torch.tensor([19.,-37.],dtype=v.dtype)
    reference=pack.dsi_peak_values(p,v,torch.zeros_like(p.cost_scales))
    shifted=pack.dsi_peak_values(p,2*v+baseline,baseline[p.entry_nodes])
    for key in reference:
        torch.testing.assert_close(reference[key],shifted[key])


@pytest.mark.parametrize('scale',[0.,-1.])
def test_no_positive_response_has_zero_finite_peak(scale):
    p=fixtures.make_pack()
    v=(fixtures.response_trace()*scale-25).requires_grad_()
    values=pack.dsi_peak_values(p,v,torch.full_like(p.cost_scales,-25))
    assert all(float(a.detach())==0 for a in values.values())
    sum(values.values()).backward()
    assert torch.isfinite(v.grad).all()


def test_mean_curves_before_peak_not_average_individual_peaks():
    p=replace(fixtures.make_pack(),entry_bs=torch.tensor([0,0,1,1]),
              entry_nodes=torch.tensor([0,1,0,1]),cost_pd_nds=torch.tensor([0,0,1,1]),
              entry_dsi_part_keys=(fixtures.KEY_A,)*4,dsi_t0s=torch.zeros(4,dtype=torch.long),
              dsi_n_ts=torch.full((4,),5,dtype=torch.long))
    v=torch.zeros((2,5,2),dtype=torch.float64)
    v[0,1,0]=10;v[0,2,1]=10
    v[1]=5
    actual=pack.dsi_peak_values(p,v,torch.zeros_like(p.cost_scales))
    assert actual[fixtures.KEY_A]==0  # Mean PD peak=5, mean ND peak=5.
    # Taking a peak per entry first would give PD=10 and DSI=1/3.


def test_common_window_excludes_peaks_outside_any_members_window():
    p=replace(fixtures.make_pack(),entry_bs=torch.tensor([0,0,1,1]),
              entry_nodes=torch.tensor([0,1,0,1]),cost_pd_nds=torch.tensor([0,0,1,1]),
              entry_dsi_part_keys=(fixtures.KEY_A,)*4,dsi_t0s=torch.tensor([0,1,0,0]),
              dsi_n_ts=torch.tensor([9,5,9,9]))
    v=torch.zeros((2,12,2),dtype=torch.float64)
    v[0,4,:]=4;v[0,8,:]=1000;v[1,:,:]=2
    result=pack.dsi_peak_values(p,v,torch.zeros_like(p.cost_scales))
    assert float(result[fixtures.KEY_A])==pytest.approx(1/3)


def test_fused_batch_offset_matches_isolated_trace():
    p=fixtures.make_pack()
    v=fixtures.response_trace()
    fused=torch.cat((torch.full_like(v,1000),v))
    a=pack.dsi_peak_values(p,v,torch.zeros_like(p.cost_scales))
    b=pack.dsi_peak_values(p,fused,torch.zeros_like(p.cost_scales),b_offset=2)
    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


@pytest.mark.parametrize('invalid',['missing','negative','overflow'])
def test_invalid_windows_reported(invalid):
    p=fixtures.make_pack()
    if invalid=='missing':p=replace(p,dsi_t0s=None)
    elif invalid=='negative':p=replace(p,dsi_t0s=p.dsi_t0s-100)
    else:p=replace(p,dsi_n_ts=p.dsi_n_ts+100)
    with pytest.raises(ValueError):
        pack.dsi_peak_values(p,fixtures.response_trace(),torch.zeros_like(p.cost_scales))


def test_incomplete_pd_nd_pair_omitted():
    p=replace(fixtures.make_pack(),cost_pd_nds=torch.zeros(4,dtype=torch.long))
    assert pack.dsi_peak_values(p,fixtures.response_trace(),torch.zeros_like(p.cost_scales))=={}


@pytest.mark.parametrize('scales',[{'mbar':1,'mbar_DSI':0},{'mbar':0,'mbar_DSI':0},{'mbar':0,'mbar_DSI':1},{'mbar':1,'mbar_DSI':0,fixtures.KEY_A:2}])
@pytest.mark.parametrize('sequential',[False,True])
def test_monitor_reports_both_independent_of_weights_with_one_forward(monkeypatch,scales,sequential):
    p=fixtures.make_pack()
    session=fixtures.make_session([p],scales=scales,sequential=sequential)
    calls=[]
    monkeypatch.setattr(optimization,'params_from_z',lambda z,_s:{'gain':z})
    monkeypatch.setattr(optimization,'gt_affine_from_pack',lambda _a,p,_s:(torch.ones_like(p.cost_scales),torch.zeros_like(p.cost_scales)))
    def forward(_s,params,_drive,_p):
        calls.append(torch.is_grad_enabled())
        return fixtures.response_trace()*params['gain']
    monkeypatch.setattr(optimization,'forward_pack',forward)
    z=torch.tensor([2.],dtype=torch.float64,requires_grad=True)
    z.grad=torch.tensor([7.],dtype=z.dtype)
    raw,peak=optimization._calc_dsi(z,session)
    assert raw==7 and peak==pytest.approx(.55)
    assert calls==[False]
    assert z.grad.item()==7 and z.item()==2
    assert session.part_cost_scales==scales
    assert optimization._fmt_dsi((raw,peak))=='  DSI: 7.0000  DSI_peak: 0.5500'


def test_no_mbar_monitor_does_no_forward(monkeypatch):
    monkeypatch.setattr(optimization,'forward_pack',lambda *_:pytest.fail('unexpected forward'))
    assert optimization._calc_dsi(torch.zeros(1),SimpleNamespace(iter_packs=lambda:iter([]))) is None
    assert optimization._fmt_dsi(None)==''
