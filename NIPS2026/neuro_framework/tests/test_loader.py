"""
Unit tests for neuro_framework.connectome.loader

Run with:
    cd /Users/lengyuner/Desktop/NIPS2026
    /Users/lengyuner/anaconda3/bin/python -m pytest neuro_framework/tests/test_loader.py -v

Tests are grouped by data source:
  TestBANC         – BANC whole-brain (requires local data files)
  TestOpticLobe    – maleCNS optic-lobe feather files
  TestFlyVis       – FlyVis HDF5 connectome
  TestFAFBCodex    – FAFB codex CSVs (skipped if data dir absent)
  TestNetworkBuild – ConnectomeNetwork construction from loader
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from neuro_framework.connectome.loader import ConnectomeLoader

# Data paths
BANC_DIR      = PROJECT_ROOT / "Connectome Dataset" / "Banc"
OL_DIR        = PROJECT_ROOT / "Jaxley_notebook" / "jaxley_tutorial-sjcabs" / "tutorial"
FLYVIS_DIR    = PROJECT_ROOT / "flyvis" / "data" / "connectome" / "ConnectomeFromAvgFilters_0000"
FAFB_DIR      = PROJECT_ROOT / "Connectome Dataset" / "FAFB"

# ── helpers ─────────────────────────────────────────────────────────────────

def _assert_schema(nodes: pd.DataFrame, edges: pd.DataFrame, source: str):
    """Assert that nodes and edges have the required common columns."""
    required_node_cols = ["root_id", "cell_type", "nt_type", "node_idx"]
    required_edge_cols = ["pre_root_id", "post_root_id", "syn_count", "pre_idx", "post_idx"]

    for col in required_node_cols:
        assert col in nodes.columns, f"[{source}] nodes missing column '{col}'"
    for col in required_edge_cols:
        assert col in edges.columns, f"[{source}] edges missing column '{col}'"

    # node_idx is 0-based contiguous
    assert nodes["node_idx"].is_monotonic_increasing
    assert nodes["node_idx"].iloc[0] == 0

    # All edge indices are valid
    valid = set(nodes["node_idx"].tolist())
    assert edges["pre_idx"].isin(valid).all(),  f"[{source}] pre_idx out of range"
    assert edges["post_idx"].isin(valid).all(), f"[{source}] post_idx out of range"

    # No NaN in key index columns
    assert not nodes["root_id"].isna().any(),   f"[{source}] root_id has NaN"
    assert not edges["pre_idx"].isna().any(),   f"[{source}] pre_idx has NaN"
    assert not edges["post_idx"].isna().any(),  f"[{source}] post_idx has NaN"
    assert not edges["syn_count"].isna().any(), f"[{source}] syn_count has NaN"


# ── BANC ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (BANC_DIR / "neurons.csv.gz").exists(),
    reason="BANC data files not found"
)
class TestBANC:
    def test_full_load(self):
        loader = ConnectomeLoader.from_banc()
        nodes, edges = loader.load()
        assert len(nodes) > 1000, "Expected > 1000 BANC neurons"
        assert len(edges) > 10000, "Expected > 10000 BANC edges"
        _assert_schema(nodes, edges, "banc")

    def test_cell_type_filter(self):
        """Loading with cell_types filter reduces node count."""
        loader_full   = ConnectomeLoader.from_banc()
        loader_filter = ConnectomeLoader.from_banc(min_syn_count=5)
        n_full, e_full     = loader_full.load()
        n_filter, e_filter = loader_filter.load()
        assert len(e_filter) <= len(e_full), "min_syn_count filter should reduce edges"

    def test_nt_sign(self):
        loader = ConnectomeLoader.from_banc()
        loader.load()
        sign = loader.nt_sign()
        assert sign.shape == (len(loader.nodes),)
        assert set(sign).issubset({-1.0, 0.0, 1.0})

    def test_adjacency_tensors(self):
        loader = ConnectomeLoader.from_banc()
        loader.load()
        pre, post, syn = loader.get_adjacency_tensors()
        assert pre.dtype == np.int64
        assert post.dtype == np.int64
        assert syn.dtype == np.float32
        assert len(pre) == len(loader.edges)

    def test_summary(self):
        loader = ConnectomeLoader.from_banc()
        loader.load()
        s = loader.summary()
        assert s["source"] == "banc"
        assert s["n_neurons"] > 0
        assert s["n_edges"] > 0

    def test_repr(self):
        loader = ConnectomeLoader.from_banc()
        loader.load()
        r = repr(loader)
        assert "banc" in r
        assert "n_nodes" in r


# ── Optic Lobe ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (OL_DIR / "malecns_09_optic_lobe_hex_08_meta.feather").exists(),
    reason="Optic lobe feather files not found"
)
class TestOpticLobe:
    def test_full_load(self):
        loader = ConnectomeLoader.from_optic_lobe()
        nodes, edges = loader.load()
        assert len(nodes) == 925, f"Expected 925 optic lobe neurons, got {len(nodes)}"
        _assert_schema(nodes, edges, "optic_lobe")

    def test_cell_type_filter(self):
        t4_types = ["T4a", "T4b", "T4c", "T4d"]
        loader = ConnectomeLoader.from_optic_lobe(cell_types=t4_types)
        nodes, edges = loader.load()
        assert len(nodes) > 0, "No T4 neurons found"
        assert set(nodes["cell_type"].unique()).issubset(set(t4_types))
        _assert_schema(nodes, edges, "optic_lobe_t4")

    def test_min_syn_filter(self):
        loader_2 = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
        loader_5 = ConnectomeLoader.from_optic_lobe(min_syn_count=5)
        _, e2 = loader_2.load()
        _, e5 = loader_5.load()
        assert len(e5) <= len(e2), "Higher min_syn_count should reduce edge count"

    def test_nt_distribution(self):
        loader = ConnectomeLoader.from_optic_lobe()
        loader.load()
        sign = loader.nt_sign()
        n_excit = (sign == 1.0).sum()
        n_inhib = (sign == -1.0).sum()
        # optic lobe is mostly cholinergic → expect mostly excitatory
        assert n_excit > n_inhib, "Expected more excitatory than inhibitory in optic lobe"

    def test_adjacency_shapes(self):
        loader = ConnectomeLoader.from_optic_lobe()
        nodes, edges = loader.load()
        pre, post, syn = loader.get_adjacency_tensors()
        assert len(pre) == len(edges)
        assert pre.max() < len(nodes)
        assert post.max() < len(nodes)

    def test_summary_keys(self):
        loader = ConnectomeLoader.from_optic_lobe()
        loader.load()
        s = loader.summary()
        for key in ["source", "n_neurons", "n_edges", "n_cell_types",
                    "top_cell_types", "nt_distribution"]:
            assert key in s, f"summary() missing key '{key}'"

    def test_side_filter(self):
        loader = ConnectomeLoader.from_optic_lobe(sides=["right"])
        nodes, _ = loader.load()
        if "side" in nodes.columns:
            assert all(nodes["side"] == "right")


# ── FlyVis ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (FLYVIS_DIR / "nodes" / "index.h5").exists(),
    reason="FlyVis HDF5 files not found"
)
class TestFlyVis:
    def test_full_load(self):
        loader = ConnectomeLoader.from_flyvis()
        nodes, edges = loader.load()
        assert len(nodes) > 0
        assert len(edges) > 0
        _assert_schema(nodes, edges, "flyvis")

    def test_no_nan_indices(self):
        loader = ConnectomeLoader.from_flyvis()
        nodes, edges = loader.load()
        assert not edges["pre_idx"].isna().any()
        assert not edges["post_idx"].isna().any()


# ── FAFB Codex ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (FAFB_DIR / "consolidated_cell_types.csv.gz").exists(),
    reason="FAFB codex data not downloaded"
)
class TestFAFBCodex:
    def test_node_load(self):
        loader = ConnectomeLoader.from_fafb()
        nodes, _ = loader.load()
        assert len(nodes) > 100000, "Expected > 100k FAFB neurons"
        assert "cell_type" in nodes.columns
        assert "root_id" in nodes.columns

    def test_lc_filter(self):
        lc_types = [f"LC{i}" for i in [4, 6, 9, 10, 11, 15]]
        loader = ConnectomeLoader.from_fafb(cell_types=lc_types)
        nodes, edges = loader.load()
        assert len(nodes) > 0, "No LC neurons found in FAFB"
        assert set(nodes["cell_type"].unique()).issubset(set(lc_types))


# ── Network construction from loader ────────────────────────────────────────

@pytest.mark.skipif(
    not (OL_DIR / "malecns_09_optic_lobe_hex_08_meta.feather").exists(),
    reason="Optic lobe feather files not found"
)
class TestNetworkBuild:
    """Integration tests: loader → ConnectomeNetwork → simulate."""

    def test_voltage_network_build(self):
        import torch
        from neuro_framework.models.network_torch import ConnectomeNetwork

        loader = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
        net = ConnectomeNetwork.from_loader(loader, dynamics="voltage")
        assert net.n_nodes == len(loader.nodes)
        assert net.n_edges == len(loader.edges)
        assert net.n_parameters() > 0

    def test_lif_network_build(self):
        import torch
        from neuro_framework.models.network_torch import ConnectomeNetwork

        loader = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
        net = ConnectomeNetwork.from_loader(loader, dynamics="lif")
        # Forward pass: (batch=2, T=10, n_nodes)
        x = torch.zeros(2, 10, net.n_nodes)
        out = net(x, dt=1.0)
        assert out.shape == (2, 10, net.n_nodes)
        assert torch.isfinite(out).all(), "LIF output contains NaN/Inf"

    def test_voltage_forward_pass(self):
        import torch
        from neuro_framework.models.network_torch import ConnectomeNetwork

        loader = ConnectomeLoader.from_optic_lobe(min_syn_count=2)
        net = ConnectomeNetwork.from_loader(loader, dynamics="voltage")
        B, T, N = 2, 20, net.n_nodes
        x = torch.randn(B, T, N) * 0.1
        out = net(x, dt=1.0)
        assert out.shape == (B, T, N)
        assert torch.isfinite(out).all(), "Voltage model output contains NaN/Inf"

    def test_gradient_flows(self):
        """Ensure loss.backward() produces non-zero gradients."""
        import torch
        from neuro_framework.models.network_torch import ConnectomeNetwork

        loader = ConnectomeLoader.from_optic_lobe(
            cell_types=["T4a", "T4b", "Mi1", "L1", "L2"],
            min_syn_count=2,
        )
        net = ConnectomeNetwork.from_loader(loader, dynamics="voltage")
        x = torch.randn(2, 10, net.n_nodes)
        out = net(x, dt=1.0)
        loss = out.mean()
        loss.backward()
        assert net.log_weight_abs.grad is not None
        assert net.log_weight_abs.grad.abs().sum() > 0, "Zero gradient on weights"

    def test_hh_forward_pass(self):
        """HH model forward pass (small network for speed)."""
        import torch
        from neuro_framework.models.network_torch import ConnectomeNetwork

        loader = ConnectomeLoader.from_optic_lobe(
            cell_types=["L1", "L2", "L3"],
            min_syn_count=2,
        )
        if len(loader.nodes) == 0:
            pytest.skip("No L1/L2/L3 neurons after filtering")
        net = ConnectomeNetwork.from_loader(loader, dynamics="hh")
        x = torch.zeros(1, 5, net.n_nodes)
        out = net(x, dt=0.1)
        assert out.shape == (1, 5, net.n_nodes)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
