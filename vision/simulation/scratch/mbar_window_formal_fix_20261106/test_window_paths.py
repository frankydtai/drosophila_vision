from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path('/home/yitai/drosophila_vision/vision/simulation')
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
import import_bootstrap
import train.cost as cost
from task.mbar.pack import MbarPack

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('zero_before',[False,True])
@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('aligned',[True,False])
@pytest.mark.parametrize('reduction',['mean_trace','entry_sse'])
def test_window_loss_and_gradient_paths(monkeypatch,dtype,zero_before,sparse,aligned,reduction):
    trace=(-60+torch.arange(56,dtype=dtype).reshape(2,14,2)/3).requires_grad_()
    drive=torch.zeros((2,14,2),dtype=dtype)
    drive[1]=1
    starts=torch.tensor([1,4,5,9]) if aligned else None
    pack=MbarPack(
        task='mbar',contrast='bright',i_sti=drive,
        gts=torch.full((4,4),-10.,dtype=dtype),cost_scales=torch.ones(4,dtype=dtype),
        entry_bs=torch.tensor([0,0,1,1]),entry_nodes=torch.tensor([0,1,0,1]),
        cost_t0s=starts,cost_ts=torch.tensor([0,2]) if sparse else None,
        entry_part_keys=tuple('mbar_case_'+str(i) for i in range(4)),
        t_onset=5,cost_zero_before_t_onset=zero_before,
    )
    session=SimpleNamespace(tasks=('mbar',),part_cost_scales={'mbar':1,'mbar_DSI':0},
        train_opts={'cost_entry_reduce':reduction,'cost_norm':'a_gt2'},
        sim_dtype=dtype,device='cpu')
    def fake_forward(session,params,i_sti,pack):
        return trace.index_select(0,i_sti[:,0,0].long())
    def fake_affine(params,pack,session,b=None):
        n=int(pack.entry_nodes.numel())
        return torch.ones(n,dtype=dtype),torch.zeros(n,dtype=dtype)
    monkeypatch.setattr(cost,'forward_pack',fake_forward)
    monkeypatch.setattr(cost,'gt_affine_from_pack',fake_affine)
    batch=cost._pack_cost_parts_from_params({},pack,session)
    separate={}
    for b in [0,1]:
        separate.update(cost._pack_cost_parts_from_params({},pack,session,b=b))
    expected={}
    support=torch.zeros_like(trace,dtype=torch.bool)
    for i,key in enumerate(pack.entry_part_keys):
        ts=(torch.arange(4)+int(starts[i])) if aligned else torch.arange(5,9)
        values=trace[pack.entry_bs[i],ts,pack.entry_nodes[i]]
        if aligned and zero_before:values=torch.where(ts<5,torch.zeros_like(values),values)
        keep=torch.tensor([0,2]) if sparse else torch.arange(4)
        expected[key]=((values[keep]+10)**2).sum()
        valid=ts[keep]>=5 if aligned and zero_before else torch.ones(len(keep),dtype=torch.bool)
        support[pack.entry_bs[i],ts[keep][valid],pack.entry_nodes[i]]=True
    assert set(batch)==set(separate)==set(expected)
    for key in expected:
        torch.testing.assert_close(batch[key],expected[key])
        torch.testing.assert_close(separate[key],expected[key])
    grad_batch=torch.autograd.grad(sum(batch.values()),trace,retain_graph=True)[0]
    grad_separate=torch.autograd.grad(sum(separate.values()),trace,retain_graph=True)[0]
    grad_expected=torch.autograd.grad(sum(expected.values()),trace)[0]
    torch.testing.assert_close(grad_batch,grad_expected)
    torch.testing.assert_close(grad_separate,grad_expected)
    assert torch.count_nonzero(grad_separate[~support])==0
    if aligned and not zero_before:
        assert torch.count_nonzero(grad_separate[:,:5])>0
