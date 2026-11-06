from types import SimpleNamespace

import pytest
import torch

import neuron.forward as neuron_forward
import neuron.point_hh as point_hh
from neuron.point_hh import build_point_hh_context


DTYPE = torch.float64


class _NoEdges:
    source_idxs = torch.empty(0, dtype=torch.long)
    target_idxs = torch.empty(0, dtype=torch.long)


class _OneEdge:
    source_idxs = torch.tensor([0])
    target_idxs = torch.tensor([1])


def _session(n_node=1, conn=None, **overrides):
    values = dict(
        connectome=SimpleNamespace(
            conn=_NoEdges() if conn is None else conn,
            n_node=n_node,
        ),
        point_hh_v_min=-120.0,
        point_hh_v_max=80.0,
        pre_steady="solve",
        pre_steady_n_iter=100,
        pre_steady_damp=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _t4a_params(n_node=1):
    # StableHH T4a densities converted using an effective area of 100 um^2.
    values = {
        "cap": 1.0,
        "gNa": 6.5,
        "gK": 2.0,
        "gLeak": 1.6,
        "eNa": 50.0,
        "eK": -80.0,
        "eLeak": -60.0,
    }
    return {
        name: torch.full((n_node,), value, dtype=DTYPE)
        for name, value in values.items()
    }


def test_gate_rate_singularities_have_analytic_values_and_finite_gradients():
    v = torch.tensor([-40.0, -55.0], dtype=DTYPE, requires_grad=True)
    alpha_m, _beta_m, _alpha_h, _beta_h, alpha_n, _beta_n = (
        point_hh.gate_rates(v)
    )

    assert alpha_m[0].item() == pytest.approx(1.0)
    assert alpha_n[1].item() == pytest.approx(0.1)

    (alpha_m.sum() + alpha_n.sum()).backward()
    assert torch.isfinite(v.grad).all()


def test_background_balance_recovers_t4a_rest_and_stays_fixed():
    session = _session()
    params = _t4a_params()
    background = torch.zeros(1, 1, 1, dtype=DTYPE)

    state = point_hh.pre_steady(session, params, 1, i_sti=background)
    assert state["v"].item() == pytest.approx(-60.42460309774924, abs=1e-10)

    next_state, component = point_hh.step(
        state,
        params,
        background[:, 0],
        session,
        delta_ms=0.025,
        return_component=True,
    )
    assert next_state["v"].item() == pytest.approx(state["v"].item(), abs=1e-12)
    assert component["current_residual"].abs().max().item() < 1e-11


def test_one_excitatory_edge_releases_and_drives_the_target_inward():
    session = _session(n_node=2, conn=_OneEdge())
    params = _t4a_params(n_node=2)
    params.update(
        {
            "gS": torch.tensor([2.0], dtype=DTYPE),
            "e_syn": torch.tensor([0.0], dtype=DTYPE),
            "v_th_syn": torch.tensor([-40.0], dtype=DTYPE),
            "delta_syn": torch.tensor([5.0], dtype=DTYPE),
            "k_minus_syn": torch.tensor([0.2], dtype=DTYPE),
        }
    )
    v = torch.tensor([[-20.0, -60.0]], dtype=DTYPE)
    state = point_hh.initial_state(v, n_edge=1)

    next_state, component = point_hh.step(
        state,
        params,
        torch.zeros_like(v),
        session,
        delta_ms=0.025,
        return_component=True,
    )

    assert next_state["s"][0, 0] > 0.0
    assert component["i_syn"][0, 0] == 0.0
    assert component["i_syn"][0, 1] > 0.0


def test_driver_step_uses_one_shared_model_step():
    session = _session()
    params = _t4a_params()
    state = point_hh.initial_state(
        torch.tensor([[-60.42460309774924]], dtype=DTYPE)
    )
    drive = torch.tensor([[20.0]], dtype=DTYPE)

    actual = point_hh.step(
        state, params, drive, session, delta_ms=1.0,
    )

    assert actual["v"].shape == state["v"].shape
    assert actual["v"].item() > state["v"].item()


def test_t4a_fixed_pulse_matches_stablehh_reference():
    session = _session()
    params = _t4a_params()
    state = point_hh.initial_state(
        torch.tensor([[-60.42460309774924]], dtype=DTYPE)
    )
    peak = state["v"].item()

    for index in range(1200):
        time_ms = index * 0.025
        drive = torch.tensor(
            [[20.0 if 20.0 <= time_ms < 30.0 else 0.0]], dtype=DTYPE,
        )
        state = point_hh.step(
            state, params, drive, session, delta_ms=0.025,
        )
        peak = max(peak, state["v"].item())

    assert peak == pytest.approx(-45.92193101667511, abs=1e-10)


def test_shared_forward_returns_batch_time_neuron_shape():
    session = _session(
        model="point_hh",
        delta_ms=0.1,
        delta_ms_pre=0.1,
        train_opts={"pre_grad": False, "filter": "none"},
    )
    stimulus = torch.zeros(2, 4, 1, dtype=DTYPE)
    stimulus[:, 1:3, 0] = 20.0
    pack = SimpleNamespace(t_onset=1, entry_nodes=torch.tensor([0]))

    output = neuron_forward.forward_nodes(
        session,
        _t4a_params(),
        nodes=pack.entry_nodes,
        i_sti=stimulus,
        pack=pack,
    )

    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()


def test_curated_context_converts_density_contacts_and_histamine(tmp_path):
    ion_rules = tmp_path / "ion.csv"
    ion_rules.write_text(
        "neuron_type,gNa,gK,gLeak,eLeak,eNa,eK\n"
        "R1-6,0.003,0.0024,0.0018,-68,50,-80\n"
        "L1,0.002,0.0009,0.0012,-33,50,-80\n"
    )
    synapse_rules = tmp_path / "synapse.csv"
    synapse_rules.write_text(
        "pre_type,post_type,nt_type,gS,e_syn,v_th,k_minus,delta,s\n"
        "R1-6,L1,Histamine,0.0006,-72,-60,0.1,3,\n"
    )
    connections = tmp_path / "connections.csv"
    connections.write_text(
        "pre_root_id,post_root_id,syn_count,nt_type\n"
        "10,20,4,ACH\n"
    )
    conn = SimpleNamespace(
        source_idxs=torch.tensor([0]),
        target_idxs=torch.tensor([1]),
        w_signed=torch.tensor([-1.0], dtype=DTYPE),
        n_edge=1,
    )
    connectome = SimpleNamespace(
        conn=conn,
        cells=["R1-6", "L1"],
        node_cells=torch.tensor([0, 1]),
        node_ids=[10, 20],
        edge_n_syn=torch.tensor([4.0], dtype=DTYPE),
        edge_source_cells=("R1-6",),
        edge_target_cells=("L1",),
        device="cpu",
        source_json=tmp_path / "network.json",
    )

    context = build_point_hh_context(
        connectome,
        ion_rules_path=ion_rules,
        synapse_rules_path=synapse_rules,
        connections_path=connections,
        area_um2=100.0,
        capacitance_uF_cm2=1.0,
    )

    assert context.params["cap"].tolist() == [1.0, 1.0]
    assert context.params["gNa"].tolist() == pytest.approx([3.0, 2.0])
    assert context.params["gS"].item() == pytest.approx(2.4)
    assert context.edge_rows[0]["raw_nt_type"] == "ACH"
    assert context.edge_rows[0]["model_nt_type"] == "Histamine"
    assert context.edge_rows[0]["resolution"] == "exact"

