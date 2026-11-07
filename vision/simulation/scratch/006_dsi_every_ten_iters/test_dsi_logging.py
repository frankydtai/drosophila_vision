from pathlib import Path
from dataclasses import replace
import importlib.util
import io
import re
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'vision'))
import import_bootstrap
import train.cost as cost
from task.mbar.pack import dsi_values

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

new = load('dsi_candidate_ten', ROOT / 'vision/simulation/4_train/4_optimization.py')
old = load('dsi_before_ten', HERE.parent / '005_dsi_training_log/4_optimization.before.py')
fixtures = load('dsi_fixtures', ROOT / 'vision/simulation/tests/test_mbar_dsi_cost.py')

@pytest.mark.parametrize('scales', [
    {'mbar': 1, 'mbar_DSI': 0}, {'mbar': 0, 'mbar_DSI': 0},
    {'mbar': 0, 'mbar_DSI': 1}, {'mbar': 1, 'mbar_DSI': 2, fixtures.KEY_A: 0},
])
@pytest.mark.parametrize('sequential', [False, True])
def test_monitor_ignores_loss_weights_and_has_no_gradient(monkeypatch, scales, sequential):
    pack = fixtures.make_pack()
    trace = fixtures.response_trace()
    session = fixtures.make_session([pack], scales=scales, sequential=sequential)
    z = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    z.grad = torch.tensor([7.0], dtype=z.dtype)
    calls = []
    monkeypatch.setattr(new, 'params_from_z', lambda z, _session: {'gain': z})
    def forward(_session, params, _drive, _pack):
        calls.append(torch.is_grad_enabled())
        return params['gain'] * trace
    monkeypatch.setattr(new, 'forward_pack', forward)
    assert new._calc_dsi(z, session) == 7.0  # (T4a=6 + T4b=8) / 2.
    assert calls == [False]
    assert z.item() == 2 and z.grad.item() == 7
    assert session.part_cost_scales == scales

@pytest.mark.parametrize('gain', [0.0, -1.0, 2.0])
def test_multiple_contrasts_and_negative_values(monkeypatch, gain):
    pack = fixtures.make_pack()
    dark = replace(pack, contrast='dark', entry_dsi_part_keys=tuple(k.replace('bright', 'dark') for k in pack.entry_dsi_part_keys))
    session = fixtures.make_session([pack, dark])
    monkeypatch.setattr(new, 'params_from_z', lambda *_: {})
    monkeypatch.setattr(new, 'forward_pack', lambda _s, _p, _i, p: fixtures.response_trace() * (gain if p.contrast == 'bright' else 2 * gain))
    assert new._calc_dsi(torch.tensor(0), session) == pytest.approx(5.25 * gain)
    assert new._fmt_dsi(new._calc_dsi(torch.tensor(0), session)) == f'  DSI: {5.25*gain:.4f}'

@pytest.mark.parametrize('packs', [[], [SimpleNamespace()]])
def test_non_mbar_session_has_no_forward(monkeypatch, packs):
    monkeypatch.setattr(new, 'forward_pack', lambda *_: pytest.fail('unexpected forward'))
    assert new._calc_dsi(torch.tensor(0), SimpleNamespace(iter_packs=lambda: iter(packs))) is None
    assert new._fmt_dsi(None) == ''


def test_incomplete_group_skipped_and_invalid_windows_reported(monkeypatch):
    pack = fixtures.make_pack()
    incomplete = replace(pack, cost_pd_nds=torch.zeros_like(pack.cost_pd_nds))
    monkeypatch.setattr(new, 'params_from_z', lambda *_: {})
    monkeypatch.setattr(new, 'forward_pack', lambda *_: fixtures.response_trace())
    assert new._calc_dsi(torch.tensor(0), fixtures.make_session([incomplete])) is None
    invalid = replace(pack, dsi_t0s=pack.dsi_t0s + 100)
    with pytest.raises(ValueError, match='outside the simulated trace'):
        new._calc_dsi(torch.tensor(0), fixtures.make_session([invalid]))


class Bar:
    records = []
    def __init__(self, values, **kwargs):
        self.values = values
        self.records.append(kwargs['desc'])
        assert kwargs['miniters'] == 10
    def __iter__(self):
        return iter(self.values)
    def set_description(self, desc, refresh):
        assert refresh is False
        self.records.append(desc)

@pytest.mark.parametrize('n_iter', [0, 1, 9, 10, 11, 25])
@pytest.mark.parametrize('sequential', [False, True])
@pytest.mark.parametrize('checkpoints', [False, True])
def test_actual_cost_gradient_adam_and_logging_unchanged(monkeypatch, capsys, n_iter, sequential, checkpoints):
    pack = replace(fixtures.make_pack(), i_sti=fixtures.response_trace())
    session = fixtures.make_session([pack], sequential=sequential, scales={'mbar': 1, 'mbar_DSI': 0})
    monkeypatch.setattr(cost, 'params_from_z', lambda z, _s: {'gain': z})
    monkeypatch.setattr(new, 'params_from_z', lambda z, _s: {'gain': z})
    def forward(_session, params, drive, _pack):
        return drive * params['gain']
    monkeypatch.setattr(cost, 'forward_pack', forward)
    monkeypatch.setattr(new, 'forward_pack', forward)
    monkeypatch.setattr(cost, 'gt_affine_from_pack', lambda _p, p, _s, **kw: (torch.ones_like(p.cost_scales), torch.zeros_like(p.cost_scales)))
    results = []
    for module in (old, new):
        monkeypatch.setattr(module, 'tqdm', Bar)
        Bar.records = []
        z = torch.tensor([1.0], dtype=torch.float64)
        fn, history, log, last, backward, evaluate = module._build_iter_logger(session)
        observed = []
        def dsi_fn(z):
            result = new._calc_dsi(z, session)
            observed.append((float(z.detach()), result))
            return result
        checkpoint_records = []
        kwargs = dict(n_iter=n_iter, device='cpu', z_clamps=torch.tensor([[-10., 10.]]), lr=.02,
                      cost_fn=fn, float_last_parts=last, task_order=list(history),
                      backward_iter=backward, eval_cost=evaluate, iter_log=log,
                      checkpoint_interval=3 if checkpoints else None,
                      on_interval_best=lambda i, z, c, **kw: checkpoint_records.append((i, float(z), c)))
        if module is new:
            kwargs['dsi_fn'] = dsi_fn
        best_z, adam = module.gradient_network(z, **kwargs)
        results.append((best_z, adam, history, checkpoint_records))
        if module is new:
            assert len(observed) == n_iter // 10 + 2
            for completed, desc in enumerate(Bar.records):
                _z, dsi = observed[completed // 10]
                assert desc.endswith(f'  DSI: {dsi:.4f}')
                assert desc.index('DSI:') > desc.index(']')
            assert all(d == pytest.approx(3.5*v) for v, d in observed)
            output = capsys.readouterr().out
            best_line = next(line for line in output.splitlines() if line.startswith('Best  cost ='))
            if 'DSI:' in best_line:
                assert best_line.endswith(f'DSI: {3.5*float(best_z):.4f}')
            (HERE / f'example_{sequential}_{n_iter}_{checkpoints}.log').write_text('\n'.join(Bar.records) + '\n' + output)
    for i in (0,):
        torch.testing.assert_close(results[0][i], results[1][i], rtol=0, atol=0)
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(results[0][1][key], results[1][1][key], rtol=0, atol=0)
    assert results[0][1]['iter'] == results[1][1]['iter']
    assert results[0][2:] == results[1][2:]


def test_staged_callback_forwarding(monkeypatch):
    calls = []
    fn = lambda z: 42
    def gradient(z, **kwargs):
        assert kwargs['dsi_fn'] is fn
        calls.append(kwargs['global_iter_start'])
        return z, {}
    monkeypatch.setattr(new, 'gradient_network', gradient)
    monkeypatch.setattr(new, 'active_device', lambda: 'cpu')
    new.optimize_staged(torch.zeros(1), None, None, [.1, .01], 3, dsi_fn=fn)
    assert calls == [0, 3]


def test_disabled_callback_and_abort_paths(monkeypatch, capsys):
    monkeypatch.setattr(new, 'tqdm', Bar)
    for fail in ('none', 'nan', 'runtime'):
        calls = [0]
        def fn(z):
            calls[0] += 1
            if calls[0] == 2 and fail == 'runtime':
                raise RuntimeError('test failure')
            return z.square().sum() * (float('nan') if calls[0] == 2 and fail == 'nan' else 1)
        new.gradient_network(torch.ones(1), n_iter=2, cost_fn=fn, z_clamps=torch.tensor([[-10.,10.]]))
        out = capsys.readouterr().out
        assert 'DSI:' not in out
        assert ('ABORT:' in out) == (fail != 'none')


def test_do_many_runs_wires_monitor_automatically(monkeypatch):
    session = SimpleNamespace(schema={}, sim_dtype=torch.float64)
    monkeypatch.setattr(new, 'schema_copy', lambda s: s)
    monkeypatch.setattr(new, 'schema_n_z', lambda s: 1)
    monkeypatch.setattr(new, 'schema_clamps', lambda *a: torch.tensor([[-10., 10.]]))
    monkeypatch.setattr(new, 'session_cost_part_keys', lambda s: ('mbar_test',))
    monkeypatch.setattr(new, '_build_iter_logger', lambda s: (None, {'mbar_test': [1.]}, lambda z: 1., None, None, None))
    def staged(z, *args, **kwargs):
        assert kwargs['dsi_fn'](z) == 42.
        return z, {'exp_avg': torch.zeros_like(z), 'exp_avg_sq': torch.zeros_like(z), 'iter': 0}
    monkeypatch.setattr(new, 'optimize_staged', staged)
    monkeypatch.setattr(new, '_calc_dsi', lambda z, s: 42. if s is session else pytest.fail('wrong session'))
    monkeypatch.setattr(new, 'calc_cost_parts', lambda z, s: {'mbar_test': torch.tensor(1.)})
    monkeypatch.setattr(new, '_scaled_cost_from_parts', lambda p, s: sum(p.values()))
    result = new.do_many_runs(session, 1, 1, z_init=torch.ones(1))
    assert result.final_costs.tolist() == [1.]
    assert list(result.costs_by_part) == ['mbar_test']


def test_no_callback_matches_original_output(monkeypatch, capsys):
    outputs = []
    for module in (old, new):
        monkeypatch.setattr(module, 'tqdm', Bar)
        module.gradient_network(torch.ones(1), n_iter=1, cost_fn=lambda z: z.square().sum(), z_clamps=torch.tensor([[-10., 10.]]))
        text = capsys.readouterr().out
        outputs.append('\n'.join(line for line in text.splitlines() if 'time needed' not in line))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('offset', [0, 3, 9, 10])
@pytest.mark.parametrize('n_iter', [0, 1, 9, 10, 11, 19, 20, 21])
def test_global_schedule_and_cached_display(monkeypatch, offset, n_iter):
    monkeypatch.setattr(new, 'tqdm', Bar)
    Bar.records = []
    cost_calls = [0]
    observations = []
    def fn(z):
        cost_calls[0] += 1
        return z.square().sum()
    def dsi_fn(z):
        observations.append(cost_calls[0])
        return float(cost_calls[0])
    new.gradient_network(torch.ones(1), n_iter=n_iter, cost_fn=fn,
                         z_clamps=torch.tensor([[-10., 10.]]),
                         dsi_fn=dsi_fn, global_iter_start=offset)
    expected = [1] + [i+1 for i in range(1,n_iter+1) if (offset+i)%10 == 0] + [n_iter+2]
    assert observations == expected
    cached = 1
    for i,desc in enumerate(Bar.records):
        if i and (offset+i)%10 == 0:
            cached = i+1
        assert desc.endswith(f'DSI: {cached:.4f}')


def test_unsampled_best_has_no_stale_dsi(monkeypatch, capsys):
    monkeypatch.setattr(new, 'tqdm', Bar)
    calls = [0]
    def fn(z):
        calls[0] += 1
        # Iter 1 is best; neither it nor final parameters were sampled during the loop.
        return z.square().sum() + {1:10., 2:-10., 3:20.}[calls[0]]
    new.gradient_network(torch.ones(1), n_iter=1, cost_fn=fn,
                         z_clamps=torch.tensor([[-10.,10.]]), dsi_fn=lambda z: 42.)
    out = capsys.readouterr().out
    best_line = next(line for line in out.splitlines() if line.startswith('Best  cost ='))
    assert 'DSI:' not in best_line
    assert 'DSI: 42.0000' in out


@pytest.mark.parametrize('error', ['nan', 'runtime'])
def test_abort_does_not_trigger_extra_dsi(monkeypatch, error):
    monkeypatch.setattr(new, 'tqdm', Bar)
    calls = [0]
    observations = []
    def fn(z):
        calls[0] += 1
        if calls[0] == 6:
            if error == 'runtime':
                raise RuntimeError('test failure')
            return z.square().sum() * float('nan')
        return z.square().sum()
    new.gradient_network(torch.ones(1), n_iter=20, cost_fn=fn,
                         z_clamps=torch.tensor([[-10.,10.]]),
                         dsi_fn=lambda z: observations.append(calls[0]))
    assert observations == [1]
