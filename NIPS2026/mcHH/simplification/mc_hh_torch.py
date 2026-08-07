"""
Multi-Compartment Hodgkin-Huxley Simulator in PyTorch
=====================================================

Loads a (simplified) SWC file, builds a compartment graph, and solves the
branched cable equation with HH channels entirely in PyTorch.

The mathematics follow Jaxley's formulation so results are comparable:
  - Truncated-cone surface areas between SWC points
  - Resistive-load based axial conductances
  - Same unit conventions  (mV, ms, μm, mS/cm², μF/cm²)

Usage
-----
    from mc_hh_torch import MultiCompartmentHH

    model = MultiCompartmentHH.from_swc("neuron.swc", ncomp=4)
    V_trace = model.simulate(I_ext, dt=0.025, T=20.0)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Morphology: SWC -> Compartment Graph
# ---------------------------------------------------------------------------

class MorphologyGraph:
    """Build a compartmental electrical model from an SWC file.
    
    Follows the same geometry conventions as Jaxley:
      area  = truncated-cone lateral surface between consecutive SWC points
      r_load = ∫ 1/(π r²) dl  (resistive load for axial coupling)
    """

    def __init__(
        self,
        n_comp: int,
        areas: np.ndarray,
        volumes: np.ndarray,
        resistive_load_in: np.ndarray,
        resistive_load_out: np.ndarray,
        radii: np.ndarray,
        lengths: np.ndarray,
        comp_edges: List[Tuple[int, int]],
        branch_parents: List[int],
        n_branches: int,
    ):
        self.n_comp = n_comp
        self.areas = areas
        self.volumes = volumes
        self.resistive_load_in = resistive_load_in
        self.resistive_load_out = resistive_load_out
        self.radii = radii
        self.lengths = lengths
        self.comp_edges = comp_edges
        self.branch_parents = branch_parents
        self.n_branches = n_branches

    @classmethod
    def from_swc(cls, swc_path: str, ncomp: int = 1,
                 min_radius: float = 0.1) -> "MorphologyGraph":
        """Load SWC, trace branches, compartmentalize."""
        nodes = _load_swc(swc_path)
        root = _find_root(nodes)
        branches, branch_parent_idx = _trace_branches(nodes, root)

        all_areas, all_volumes = [], []
        all_r_in, all_r_out = [], []
        all_radii, all_lengths = [], []
        comp_edges: List[Tuple[int, int]] = []
        comp_offset = 0

        for b_idx, branch_node_ids in enumerate(branches):
            xyzr = np.array([[nodes[nid]['x'], nodes[nid]['y'],
                              nodes[nid]['z'], max(nodes[nid]['r'], min_radius)]
                             for nid in branch_node_ids])

            # Split branch into ncomp equal-length segments
            xyzr_per_comp = _split_xyzr(xyzr, ncomp)

            for c_idx, comp_xyzr in enumerate(xyzr_per_comp):
                r, a, v, rl_in, rl_out = _comp_morph_attrs(comp_xyzr, min_radius, ncomp)
                all_radii.append(r)
                all_areas.append(a)
                all_volumes.append(v)
                all_r_in.append(rl_in)
                all_r_out.append(rl_out)

                positions = comp_xyzr[:, :3]
                if len(positions) > 1:
                    seg_len = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                else:
                    seg_len = 2 * comp_xyzr[0, 3] / ncomp
                all_lengths.append(seg_len)

            # Intra-branch edges (comp i <-> comp i+1)
            for c in range(ncomp - 1):
                ci = comp_offset + c
                cj = comp_offset + c + 1
                comp_edges.append((ci, cj))
                comp_edges.append((cj, ci))

            # Inter-branch edge: parent branch's last comp <-> this branch's first comp
            parent_b = branch_parent_idx[b_idx]
            if parent_b >= 0:
                parent_last_comp = parent_b * ncomp + (ncomp - 1)
                child_first_comp = comp_offset
                comp_edges.append((parent_last_comp, child_first_comp))
                comp_edges.append((child_first_comp, parent_last_comp))

            comp_offset += ncomp

        return cls(
            n_comp=comp_offset,
            areas=np.array(all_areas),
            volumes=np.array(all_volumes),
            resistive_load_in=np.array(all_r_in),
            resistive_load_out=np.array(all_r_out),
            radii=np.array(all_radii),
            lengths=np.array(all_lengths),
            comp_edges=comp_edges,
            branch_parents=branch_parent_idx,
            n_branches=len(branches),
        )


# ---------------------------------------------------------------------------
# Multi-Compartment HH Model (PyTorch)
# ---------------------------------------------------------------------------

class MultiCompartmentHH(nn.Module):
    """
    Multi-compartment Hodgkin-Huxley neuron model in PyTorch.

    Solves the branched cable equation:

        C_m · dV_i/dt = −I_ion,i + (1/A_i) Σ_j G_ij(V_j − V_i) + I_ext,i

    where G_ij = 1 / (R_a · (r_load_out,i + r_load_in,j)) is the axial
    conductance between compartments i and j (converted to mS/cm² by
    dividing by area and capacitance, ×10⁷ for unit conversion).

    Parameters
    ----------
    morph : MorphologyGraph
        Morphology built from an SWC file.
    """

    def __init__(self, morph: MorphologyGraph):
        super().__init__()
        self.n_comp = morph.n_comp
        self.n_branches = morph.n_branches

        # Morphology buffers (not trainable by default)
        self.register_buffer("areas",  torch.tensor(morph.areas, dtype=torch.float64))
        self.register_buffer("volumes", torch.tensor(morph.volumes, dtype=torch.float64))
        self.register_buffer("r_load_in", torch.tensor(morph.resistive_load_in, dtype=torch.float64))
        self.register_buffer("r_load_out", torch.tensor(morph.resistive_load_out, dtype=torch.float64))
        self.register_buffer("radii", torch.tensor(morph.radii, dtype=torch.float64))
        self.register_buffer("lengths", torch.tensor(morph.lengths, dtype=torch.float64))

        # Build sparse axial-conductance matrix
        G_sparse = self._build_axial_matrix(morph)
        self.register_buffer("G_row", G_sparse[0])
        self.register_buffer("G_col", G_sparse[1])
        self.register_buffer("G_val", G_sparse[2])

        # HH parameters (trainable)
        self.log_g_Na = nn.Parameter(torch.tensor(math.log(120.0)))
        self.log_g_K  = nn.Parameter(torch.tensor(math.log(36.0)))
        self.log_g_L  = nn.Parameter(torch.tensor(math.log(0.3)))
        self.E_Na = nn.Parameter(torch.tensor(50.0))
        self.E_K  = nn.Parameter(torch.tensor(-77.0))
        self.E_L  = nn.Parameter(torch.tensor(-54.4))
        self.log_C_m = nn.Parameter(torch.tensor(math.log(1.0)))

        # Axial resistivity (Ω·cm), trainable
        self.log_R_a = nn.Parameter(torch.tensor(math.log(100.0)))

    # ------------------------------------------------------------------
    @classmethod
    def from_swc(cls, swc_path: str, ncomp: int = 1,
                 min_radius: float = 0.1) -> "MultiCompartmentHH":
        morph = MorphologyGraph.from_swc(swc_path, ncomp=ncomp,
                                         min_radius=min_radius)
        return cls(morph)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def g_Na(self) -> Tensor: return torch.exp(self.log_g_Na)
    @property
    def g_K(self) -> Tensor:  return torch.exp(self.log_g_K)
    @property
    def g_L(self) -> Tensor:  return torch.exp(self.log_g_L)
    @property
    def C_m(self) -> Tensor:  return torch.exp(self.log_C_m)
    @property
    def R_a(self) -> Tensor:  return torch.exp(self.log_R_a)

    # ------------------------------------------------------------------
    # Axial conductance matrix
    # ------------------------------------------------------------------
    def _build_axial_matrix(self, morph: MorphologyGraph):
        """Build COO sparse representation of the axial conductance matrix.
        
        The matrix encodes: for each directed edge (i→j), the axial conductance
        divided by (C_m · A_j) is stored. The conductance depends on R_a which
        is trainable, so we store the *geometric* part and multiply R_a at runtime.
        """
        rows, cols, geom_vals = [], [], []

        for (src, snk) in morph.comp_edges:
            # resistive load from source's output side + sink's input side
            r_load = morph.resistive_load_out[src] + morph.resistive_load_in[snk]
            if r_load < 1e-30:
                r_load = 1e-30
            # geometric conductance factor: 1/r_load (will be divided by R_a at runtime)
            geom = 1.0 / r_load
            rows.append(snk)  # current flows INTO snk
            cols.append(src)
            geom_vals.append(geom)

        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(geom_vals, dtype=torch.float64),
        )

    def _build_coupling_matrix(self) -> Tensor:
        """Build the dense axial coupling matrix G such that
        
            G @ V  gives the axial dV/dt contribution (mV/ms) for each compartment.
        
        G[i,j] = g_ij / (C_m * A_i) * 1e7    for j ≠ i  (off-diagonal)
        G[i,i] = -Σ_{j≠i} G[i,j]             (diagonal, so rows sum to 0)
        
        Unit derivation (Jaxley convention):
            g_ij = 1 / (R_a * (r_load_out_i + r_load_in_j))
            R_a in Ω·cm, r_load in μm⁻¹, A in μm², C_m in μF/cm²
            The 1e7 factor converts to mV/ms.
        """
        N = self.n_comp
        G = torch.zeros(N, N, dtype=torch.float64, device=self.areas.device)

        g_axial = self.G_val / self.R_a  # 1/(R_a * r_load)
        norm = 1e7 / (self.C_m * self.areas)  # (N,)

        for k in range(len(self.G_val)):
            i = self.G_row[k].item()  # sink
            j = self.G_col[k].item()  # source
            val = g_axial[k] * norm[i]
            G[i, j] = G[i, j] + val
            G[i, i] = G[i, i] - val

        return G

    # ------------------------------------------------------------------
    # HH gating kinetics
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_x_over_1_minus_exp_neg_x(x: Tensor) -> Tensor:
        """Compute x / (1 - exp(-x)) in a numerically stable way.
        
        Near x=0 this equals 1 (by L'Hôpital); the naive formula gives 0/0.
        """
        safe = torch.where(
            torch.abs(x) < 1e-6,
            torch.ones_like(x),
            x / (1.0 - torch.exp(-x))
        )
        return safe

    @staticmethod
    def _alpha_m(V):
        # α_m = 0.1*(V+40) / (1 - exp(-(V+40)/10))
        x = (V + 40.0) / 10.0
        return 0.1 * 10.0 * MultiCompartmentHH._safe_x_over_1_minus_exp_neg_x(x)

    @staticmethod
    def _beta_m(V):
        return 4.0 * torch.exp(-(V + 65.0) / 18.0)

    @staticmethod
    def _alpha_h(V):
        return 0.07 * torch.exp(-(V + 65.0) / 20.0)

    @staticmethod
    def _beta_h(V):
        return 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))

    @staticmethod
    def _alpha_n(V):
        # α_n = 0.01*(V+55) / (1 - exp(-(V+55)/10))
        x = (V + 55.0) / 10.0
        return 0.01 * 10.0 * MultiCompartmentHH._safe_x_over_1_minus_exp_neg_x(x)

    @staticmethod
    def _beta_n(V):
        return 0.125 * torch.exp(-(V + 65.0) / 80.0)

    def _m_inf(self, V):
        a = self._alpha_m(V); return a / (a + self._beta_m(V))

    def _h_inf(self, V):
        a = self._alpha_h(V); return a / (a + self._beta_h(V))

    def _n_inf(self, V):
        a = self._alpha_n(V); return a / (a + self._beta_n(V))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int = 1,
                   device: torch.device = torch.device("cpu")) -> Dict[str, Tensor]:
        V0 = torch.full((batch_size, self.n_comp), -65.0,
                        dtype=torch.float64, device=device)
        return {
            "V": V0,
            "m": self._m_inf(V0),
            "h": self._h_inf(V0),
            "n": self._n_inf(V0),
        }

    # ------------------------------------------------------------------
    # Single time step  (semi-implicit: implicit axial, explicit channels)
    # ------------------------------------------------------------------
    def step(self, state: Dict[str, Tensor], I_ext: Tensor,
             dt: float = 0.025) -> Dict[str, Tensor]:
        """Advance one time step.
        
        Uses a semi-implicit (Strang-split) scheme matching Jaxley's default:
          - Channel / gating kinetics: explicit (forward Euler)
          - Axial coupling: implicit (backward Euler)
        
        This is unconditionally stable for the diffusive (axial) part,
        allowing dt = 0.025 ms even for very fine morphologies.
        
        The implicit step solves:
            (I - dt·G) V^{n+1} = V^n + dt·f(V^n)
        where G is the axial coupling matrix and f contains membrane
        currents + external input.
        
        Args:
            state: dict with keys V, m, h, n  — each (batch, n_comp)
            I_ext: external current density (μA/cm²), shape (batch, n_comp)
            dt: time step in ms
            
        Returns:
            Updated state dict.
        """
        V = state["V"]
        m, h, n = state["m"], state["h"], state["n"]

        # --- 1. Membrane currents (explicit, evaluated at V^n) ---
        # voltage_terms = total membrane conductance / C_m  [1/ms]
        g_total = self.g_Na * m**3 * h + self.g_K * n**4 + self.g_L
        voltage_terms = g_total / self.C_m

        # constant_terms = (Σ g_k E_k + I_ext) / C_m  [mV/ms]
        constant_terms = (
            self.g_Na * m**3 * h * self.E_Na
            + self.g_K * n**4 * self.E_K
            + self.g_L * self.E_L
            + I_ext
        ) / self.C_m

        # --- 2. Build and solve implicit system for voltage ---
        # dV/dt = -voltage_terms * V + constant_terms + G @ V
        # Backward Euler for axial part:
        #   (I + dt*diag(voltage_terms) - dt*G) V^{n+1} = V^n + dt*constant_terms
        G = self._build_coupling_matrix()  # (N, N)

        N = self.n_comp
        # A = I + dt*(diag(voltage_terms) - G)
        A = -dt * G
        # Add diagonal terms
        diag_add = 1.0 + dt * voltage_terms  # (B, N)
        # A shape needs to be (B, N, N) for batched solve
        B_size = V.shape[0]
        A_batch = A.unsqueeze(0).expand(B_size, -1, -1).clone()
        for i in range(N):
            A_batch[:, i, i] = A_batch[:, i, i] + diag_add[:, i]

        rhs = V + dt * constant_terms  # (B, N)

        V_new = torch.linalg.solve(A_batch, rhs.unsqueeze(-1)).squeeze(-1)

        # --- 3. Gating variables (explicit Euler) ---
        am, bm = self._alpha_m(V), self._beta_m(V)
        ah, bh = self._alpha_h(V), self._beta_h(V)
        an, bn = self._alpha_n(V), self._beta_n(V)

        dm = am * (1.0 - m) - bm * m
        dh = ah * (1.0 - h) - bh * h
        dn = an * (1.0 - n) - bn * n

        return {
            "V": V_new,
            "m": torch.clamp(m + dt * dm, 0.0, 1.0),
            "h": torch.clamp(h + dt * dh, 0.0, 1.0),
            "n": torch.clamp(n + dt * dn, 0.0, 1.0),
        }

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------
    def simulate(
        self,
        I_ext: Tensor,
        dt: float = 0.025,
        state: Optional[Dict[str, Tensor]] = None,
        record_comp_indices: Optional[List[int]] = None,
    ) -> Tensor:
        """Run simulation over T time steps.
        
        Args:
            I_ext: External current, shape (batch, T, n_comp) or (T, n_comp).
                   Units: μA/cm².
            dt: Time step in ms.
            state: Optional initial state.
            record_comp_indices: Which compartments to record. If None, record all.
            
        Returns:
            V_trace: Voltage trace, shape (batch, T, n_recorded).
        """
        if I_ext.dim() == 2:
            I_ext = I_ext.unsqueeze(0)  # add batch dim

        B, T, N = I_ext.shape
        assert N == self.n_comp, (
            f"I_ext has {N} compartments but model has {self.n_comp}")

        if state is None:
            state = self.init_state(B, device=I_ext.device)

        if record_comp_indices is None:
            record_comp_indices = list(range(self.n_comp))
        rec_idx = torch.tensor(record_comp_indices, dtype=torch.long,
                               device=I_ext.device)

        V_traces = []
        for t in range(T):
            state = self.step(state, I_ext[:, t, :], dt)
            V_traces.append(state["V"][:, rec_idx])

        return torch.stack(V_traces, dim=1)  # (B, T, n_recorded)

    def forward(self, I_ext: Tensor, dt: float = 0.025, **kw) -> Tensor:
        return self.simulate(I_ext, dt=dt, **kw)

    def extra_repr(self) -> str:
        return (f"n_comp={self.n_comp}, n_branches={self.n_branches}, "
                f"n_edges={len(self.G_val)}")


# ---------------------------------------------------------------------------
# SWC geometry helpers (matching Jaxley conventions)
# ---------------------------------------------------------------------------

def _load_swc(path: str) -> Dict[int, Dict]:
    nodes = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            nid = int(parts[0])
            nodes[nid] = {
                "id": nid, "type": int(parts[1]),
                "x": float(parts[2]), "y": float(parts[3]), "z": float(parts[4]),
                "r": float(parts[5]), "parent": int(parts[6]), "children": [],
            }
    for nid, n in nodes.items():
        pid = n["parent"]
        if pid != -1 and pid in nodes:
            nodes[pid]["children"].append(nid)
    return nodes


def _find_root(nodes: Dict) -> int:
    roots = [nid for nid, n in nodes.items() if n["parent"] == -1]
    return roots[0]


def _trace_branches(nodes: Dict, root: int) -> Tuple[List[List[int]], List[int]]:
    """DFS-based branch tracing.
    
    Returns:
        (branches, parent_branch_idx):
          branches[i] = list of node ids in branch i
          parent_branch_idx[i] = index of parent branch (-1 for root branch)
    
    A branch ends at a leaf or at a node with >1 children (branchpoint).
    Child branches start from the branchpoint (included as first node).
    """
    branches: List[List[int]] = []
    parent_branch_idx: List[int] = []
    # Stack items: (node_id, current_branch_nodes, parent_branch_index)
    stack: List[Tuple[int, List[int], int]] = [(root, [root], -1)]

    while stack:
        nid, current_branch, parent_bidx = stack.pop()
        children = nodes[nid]["children"]

        if len(children) == 0:
            branches.append(current_branch)
            parent_branch_idx.append(parent_bidx)
        elif len(children) == 1:
            stack.append((children[0], current_branch + [children[0]], parent_bidx))
        else:
            # Branchpoint: close current branch
            my_bidx = len(branches)
            branches.append(current_branch)
            parent_branch_idx.append(parent_bidx)
            # Start child branches from this branchpoint
            for cid in children:
                stack.append((cid, [nid, cid], my_bidx))

    return branches, parent_branch_idx


def _split_xyzr(xyzr: np.ndarray, ncomp: int) -> List[np.ndarray]:
    """Split branch xyzr into ncomp equal-length segments.
    
    Matches Jaxley's `split_xyzr_into_equal_length_segments`.
    """
    if len(xyzr) <= 1:
        return [xyzr] * ncomp

    xyz = xyzr[:, :3]
    dists = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dists = np.concatenate([[0], np.cumsum(dists)])
    total_length = cum_dists[-1]

    if total_length < 1e-10:
        return [xyzr] * ncomp

    target_dists = np.linspace(0, total_length, ncomp + 1)
    idxs = np.clip(np.searchsorted(cum_dists, target_dists, side="right") - 1,
                   0, len(xyz) - 2)
    local_dist = target_dists - cum_dists[idxs]
    segment_lens = np.where(dists[idxs] < 1e-14, 1e-14, dists[idxs])
    frac = (local_dist / segment_lens)[:, None]

    split_points = xyzr[idxs] + frac * (xyzr[idxs + 1] - xyzr[idxs])

    segments = []
    for i in range(ncomp):
        mask = (cum_dists > target_dists[i]) & (cum_dists < target_dists[i + 1])
        between = xyzr[mask]
        seg = np.vstack([split_points[i], *between, split_points[i + 1]])
        segments.append(seg)

    return segments


def _comp_morph_attrs(xyzr: np.ndarray, min_radius: float,
                      ncomp: int) -> Tuple[float, float, float, float, float]:
    """Compute radius, area, volume, r_load_in, r_load_out for one compartment.
    
    Matches Jaxley's `morph_attrs_from_xyzr`.
    """
    positions = xyzr[:, :3]
    radii = xyzr[:, 3]

    if len(xyzr) > 1:
        seg_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)

        # Weighted average radius
        radius_weights = np.zeros(len(seg_lengths) + 1)
        radius_weights[1:] += seg_lengths
        radius_weights[:-1] += seg_lengths
        radius_weights /= np.sum(radius_weights) + 1e-30
        avg_radius = np.sum(radii * radius_weights)

        # Truncated-cone surface area
        r_start, r_end = radii[:-1], radii[1:]
        delta_r = r_end - r_start
        slant = np.sqrt(delta_r**2 + seg_lengths**2)
        area = np.sum(np.pi * (r_start + r_end) * slant)

        # Truncated-cone volume
        volume = np.sum((np.pi / 3) * seg_lengths *
                        (r_start**2 + r_start * r_end + r_end**2))

        # Resistive loads (split compartment in half)
        xyzr_halves = _split_xyzr(xyzr, 2)
        r_loads = []
        for half in xyzr_halves:
            p = half[:, :3]
            r = half[:, 3]
            if len(p) > 1:
                sl = np.linalg.norm(np.diff(p, axis=0), axis=1)
                r_loads.append(_resistive_load(sl, r))
            else:
                length = r[0] / ncomp
                r_loads.append(length / r[0]**2 / np.pi)
        r_load_in, r_load_out = r_loads[0], r_loads[1]
    else:
        avg_radius = radii.mean()
        area = 4 * np.pi * radii[0]**2 / ncomp
        volume = 4 / 3 * np.pi * radii[0]**3 / ncomp
        length = radii[0] / ncomp
        r_load_in = r_load_out = length / radii[0]**2 / np.pi

    avg_radius = max(avg_radius, min_radius)
    return avg_radius, area, volume, r_load_in, r_load_out


def _resistive_load(lengths: np.ndarray, radii: np.ndarray) -> float:
    """Compute resistive load: (1/π) ∫ 1/r² dl.
    
    Matches Jaxley's `swc_resistive_load`.
    """
    r_start = radii[:-1]
    r_end = radii[1:]
    delta_r = r_end - r_start

    integrals = np.empty_like(lengths)

    constant_mask = np.isclose(delta_r, 0)
    integrals[constant_mask] = lengths[constant_mask] / r_start[constant_mask]**2

    varying_mask = ~constant_mask
    integrals[varying_mask] = (
        lengths[varying_mask] / delta_r[varying_mask]
        * (1.0 / r_start[varying_mask] - 1.0 / r_end[varying_mask])
    )

    return float(np.sum(integrals) / np.pi)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def _quick_test():
    """Smoke test with a synthetic 3-point SWC."""
    import tempfile, os

    swc_content = """\
# minimal test neuron
1 1 0.0 0.0 0.0 2.0 -1
2 3 10.0 0.0 0.0 1.5 1
3 3 20.0 0.0 0.0 1.0 2
4 3 10.0 5.0 0.0 1.5 1
5 3 10.0 10.0 0.0 1.0 4
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.swc', delete=False) as f:
        f.write(swc_content)
        tmp_path = f.name

    try:
        model = MultiCompartmentHH.from_swc(tmp_path, ncomp=2)
        print(f"Model: {model}")
        print(f"  n_comp = {model.n_comp}")
        print(f"  n_branches = {model.n_branches}")
        print(f"  n_edges = {len(model.G_val)}")
        print(f"  areas = {model.areas[:6].numpy()}")

        # Simulate 5ms with current injection at soma (comp 0)
        T_steps = 200
        dt = 0.025
        I_ext = torch.zeros(1, T_steps, model.n_comp, dtype=torch.float64)
        I_ext[0, 40:160, 0] = 15.0  # inject 15 μA/cm² into soma

        with torch.no_grad():
            V = model.simulate(I_ext, dt=dt)
        
        print(f"\n  V shape = {V.shape}")
        print(f"  V[soma] range  = [{V[0,:,0].min():.1f}, {V[0,:,0].max():.1f}] mV")
        if model.n_comp > 1:
            print(f"  V[comp1] range = [{V[0,:,1].min():.1f}, {V[0,:,1].max():.1f}] mV")
        print("\nQuick test passed!")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    _quick_test()
