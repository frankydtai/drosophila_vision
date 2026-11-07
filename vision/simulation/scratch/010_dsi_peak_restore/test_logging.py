from dataclasses import replace
import pytest
import torch
import train.cost as cost
from conftest import optimization as new, before, fixtures, HERE

class Bar:
    records=[]
    def __init__(self,values,**kwargs):
        self.values=values
        self.records.append(kwargs['desc'])
        assert kwargs['miniters']==10
    def __iter__(self):return iter(self.values)
    def set_description(self,desc,refresh):
        assert refresh is False
        self.records.append(desc)

@pytest.mark.parametrize('n_iter',[0,1,9,10,11,25])
@pytest.mark.parametrize('sequential',[False,True])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_both_metrics_preserve_cost_gradient_adam_checkpoint_and_cadence(monkeypatch,capsys,n_iter,sequential,checkpoint):
    p=replace(fixtures.make_pack(),i_sti=fixtures.response_trace())
    session=fixtures.make_session([p],sequential=sequential,scales={'mbar':1,'mbar_DSI':0})
    affine=lambda _params,p,_session,**kw:(torch.ones_like(p.cost_scales),torch.zeros_like(p.cost_scales))
    monkeypatch.setattr(cost,'params_from_z',lambda z,_s:{'gain':z})
    monkeypatch.setattr(new,'params_from_z',lambda z,_s:{'gain':z})
    monkeypatch.setattr(cost,'gt_affine_from_pack',affine)
    monkeypatch.setattr(new,'gt_affine_from_pack',affine)
    forward=lambda _s,params,drive,_p:drive*params['gain']
    monkeypatch.setattr(cost,'forward_pack',forward)
    monkeypatch.setattr(new,'forward_pack',forward)
    results=[]
    for module in (before,new):
        monkeypatch.setattr(module,'tqdm',Bar)
        Bar.records=[]
        fn,history,log,last,backward,evaluate=module._build_iter_logger(session)
        observations=[]
        def monitor(z):
            values=new._calc_dsi(z,session)
            observations.append((float(z.detach()),values))
            return values
        checkpoints=[]
        kwargs=dict(n_iter=n_iter,device='cpu',z_clamps=torch.tensor([[-10.,10.]]),lr=.02,
                    cost_fn=fn,float_last_parts=last,task_order=list(history),
                    backward_iter=backward,eval_cost=evaluate,iter_log=log,
                    checkpoint_interval=3 if checkpoint else None,
                    on_interval_best=lambda i,z,c,**kw:checkpoints.append((i,float(z.detach()),c)))
        if module is new:kwargs['dsi_fn']=monitor
        z,adam=module.gradient_network(torch.ones(1,dtype=torch.float64),**kwargs)
        results.append((z,adam,history,checkpoints))
        if module is new:
            assert len(observations)==n_iter//10+2
            for completed,desc in enumerate(Bar.records):
                _,values=observations[completed//10]
                assert desc.endswith(new._fmt_dsi(values))
                assert desc.index('DSI:')>desc.index(']')
            for gain,(raw,peak) in observations:
                assert raw==pytest.approx(3.5*gain)
                assert peak==pytest.approx(.55)
            out=capsys.readouterr().out
            (HERE/f'log_seq_{sequential}_n_{n_iter}_checkpoint_{checkpoint}.txt').write_text('\n'.join(Bar.records)+'\n'+out)
            best_line=next(line for line in out.splitlines() if line.startswith('Best  cost ='))
            if 'DSI:' in best_line:
                assert f'DSI: {3.5*float(z):.4f}' in best_line
                assert 'DSI_peak: 0.5500' in best_line
    torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
    for key in ('exp_avg','exp_avg_sq'):
        torch.testing.assert_close(results[0][1][key],results[1][1][key],rtol=0,atol=0)
    assert results[0][1]['iter']==results[1][1]['iter']
    assert results[0][2:]==results[1][2:]


@pytest.mark.parametrize('offset',[0,3,9,10])
@pytest.mark.parametrize('n_iter',[0,1,9,10,11,19,20,21])
def test_global_ten_iteration_schedule_and_cached_pair(monkeypatch,offset,n_iter):
    monkeypatch.setattr(new,'tqdm',Bar)
    Bar.records=[]
    calls=[0];observations=[]
    def fn(z):
        calls[0]+=1
        return z.square().sum()
    def monitor(z):
        observations.append(calls[0])
        return float(calls[0]),.25
    new.gradient_network(torch.ones(1),n_iter=n_iter,cost_fn=fn,
                         z_clamps=torch.tensor([[-10.,10.]]),dsi_fn=monitor,global_iter_start=offset)
    assert observations==[1]+[i+1 for i in range(1,n_iter+1) if (offset+i)%10==0]+[n_iter+2]
    cached=1
    for i,desc in enumerate(Bar.records):
        if i and (offset+i)%10==0:cached=i+1
        assert desc.endswith(f'DSI: {cached:.4f}  DSI_peak: 0.2500')


@pytest.mark.parametrize('kind',['none','nan','runtime'])
def test_abort_or_missing_callback_unchanged(monkeypatch,capsys,kind):
    monkeypatch.setattr(new,'tqdm',Bar)
    calls=[0]
    def fn(z):
        calls[0]+=1
        if calls[0]==2 and kind=='runtime':raise RuntimeError('test failure')
        return z.square().sum()*(float('nan') if calls[0]==2 and kind=='nan' else 1)
    new.gradient_network(torch.ones(1),n_iter=2,cost_fn=fn,z_clamps=torch.tensor([[-10.,10.]]))
    out=capsys.readouterr().out
    assert 'DSI:' not in out and 'DSI_peak:' not in out
    assert ('ABORT:' in out)==(kind!='none')
