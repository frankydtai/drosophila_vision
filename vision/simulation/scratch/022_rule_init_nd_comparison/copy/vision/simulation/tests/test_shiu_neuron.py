from types import SimpleNamespace

import pytest
import torch

import neuron.shiu as shiu


class _TwoNodeConn:
    """Four excitatory anatomical synapses from node 0 to node 1."""

    n_node = 2

    def exc_inh_g(self, output, _syn_strength):
        g_exc = torch.zeros_like(output)
        g_exc[..., 1] = output[..., 0] * 4.0 * 0.001
        return g_exc, torch.zeros_like(output)


def _session():
    return SimpleNamespace(
        connectome=SimpleNamespace(conn=_TwoNodeConn(), n_node=2),
        a_syn_exc=0.001,
        a_syn_inh=0.001,
        g_leak=1.0,
        euler="implicit",
        pre_steady="solve",
    )


def _params():
    return {
        "a_in": torch.ones(2),
        "syn_strength_cell": torch.ones(1),
    }


def test_shiu_rest_is_stable():
    session = _session()
    drive = torch.zeros(1, 1, 2)
    g, refractory_ms, v = shiu.pre_steady(session, _params(), 1, drive)
    g, refractory_ms, v = shiu.step(
        g, refractory_ms, v, _params(), drive[:, 0], session, delta_ms=2.0,
    )

    assert torch.equal(g, torch.zeros_like(g))
    assert torch.equal(refractory_ms, torch.zeros_like(refractory_ms))
    assert torch.equal(v, torch.full_like(v, shiu.V_REST_MV))


def test_shiu_spike_reaches_downstream_after_one_step():
    session = _session()
    params = _params()
    drive = torch.zeros(1, 1, 2)
    g, refractory_ms, v = shiu.pre_steady(session, params, 1, drive)

    # A strong current makes node 0 cross threshold.
    i_sti = torch.tensor([[100.0, 0.0]])
    g, refractory_ms, v = shiu.step(
        g, refractory_ms, v, params, i_sti, session, delta_ms=2.0,
    )
    assert v[0, 0] > shiu.V_THRESHOLD_MV
    assert refractory_ms[0, 0] == pytest.approx(shiu.T_REFRACTORY_MS)

    # Four anatomical synapses each add the paper's fixed 0.275 mV.
    g, refractory_ms, v = shiu.step(
        g, refractory_ms, v, params, torch.zeros_like(i_sti), session,
        delta_ms=2.0,
    )
    assert g[0, 1] == pytest.approx(4.0 * shiu.W_SYNAPSE_MV)
    assert v[0, 0] == pytest.approx(shiu.V_RESET_MV)
