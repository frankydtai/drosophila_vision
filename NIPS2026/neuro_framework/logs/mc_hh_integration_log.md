# Multi-Compartment HH Integration Log

**Date**: 2026-04-08

## Summary

Integrated multi-compartment Hodgkin-Huxley neuron models into `neuro_framework`, with support for real SWC morphologies (12k+ compartments) via a memory-efficient sparse Jacobi solver.

## Problem

The initial implementation used a **dense coupling matrix** `(C, C)` and `torch.linalg.solve` for the implicit axial coupling step. For a real neuron morphology (`ct1_10009.swc`, 43,742 SWC nodes → 12,563 branches), this required:

- Dense matrix: `12563 × 12563 × 8 bytes = 1.26 GB` (float64)
- Batched matrix for `torch.linalg.solve`: even larger
- O(C³) solve time per timestep

This caused out-of-memory crashes.

## Solution: Adaptive Dense/Sparse Solver

Added `_solve_implicit()` function in `mc_hh.py` that automatically selects the solver based on compartment count:

| Compartments | Solver | Memory | Time per step |
|---|---|---|---|
| ≤ 500 | Dense `torch.linalg.solve` | O(C²) | O(C³) |
| > 500 | Sparse Jacobi iteration | O(n_edges) | O(n_iter × n_edges) |

### Sparse Jacobi Solver

The semi-implicit system `(I + dt·diag(vt) − dt·G) V = rhs` has a diagonally dominant matrix, guaranteeing Jacobi convergence. The implementation:

1. Stores coupling graph as sparse COO tensor (only edge values, ~25k entries vs ~158M dense)
2. Separates diagonal from off-diagonal for efficient updates
3. Uses `torch.sparse.mm` for the matrix-vector product
4. Fixed iteration count (20) for clean autograd differentiation

### Key implementation details

- Off-diagonal entries: `a_ij = -dt × g_ax[k] × norm[i]` where `norm[i] = 1e7 / (C_m × area[i])`
- Diagonal correction: `diag_corr[i] = Σ_k(-a_ik)` ensures row-sum consistency
- Full diagonal: `a_ii = 1 + dt × vt[i] + diag_corr[i]`

## Files Modified

| File | Change |
|---|---|
| `models/mc_hh.py` | Added `_solve_implicit()` with dense/sparse paths; refactored `MultiCompartmentHH.step()` and `MCNetwork._step()` to use it |
| `models/synapses.py` | Added `NMDASynapse`, `GABAaSynapse` |
| `models/morphology.py` | New: SWC loading and compartmentalisation |
| `models/__init__.py` | Exports updated |
| `notebooks/06_mc_hh_network.ipynb` | Demo notebook with real SWC support |

## Test Results

```
=== Small Neuron (dense, 12 compartments) ===
  Soma V: [-75.6, 38.5] mV  ✓ action potentials

=== Synapse Models ===
  Ionotropic (E=0):   N0 spikes, N1 depolarised  ✓
  NMDA (E=0):         N0 spikes, N1 depolarised  ✓
  GABA_A (E=-80):     N0 spikes, N1 hyperpolarised  ✓

=== Training ===
  Gradients flow through all parameters  ✓
  TorchTrainer compatible  ✓

=== Real Neuron (sparse, 12,563 compartments) ===
  Load time: 2.5s
  Sim time (400 steps): 2.8s
  Soma V: [-61.4, 39.8] mV  ✓ action potentials
  Memory: ~25k sparse entries vs 158M dense  ✓
```

## Architecture

```
_solve_implicit()
├── if C ≤ 500:  dense torch.linalg.solve
└── if C > 500:  sparse Jacobi iteration
      ├── torch.sparse_coo_tensor (off-diagonal)
      ├── d_inv = 1 / a_diag  (diagonal preconditioner)
      └── 20 iterations: x = d_inv * (rhs - sparse_mm(off, x))
```
