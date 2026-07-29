# -*- coding: utf-8 -*-
"""Runtime dataclasses for one training / plotting run.

Leaf of the ``training`` package: depends only on :mod:`config`, the
``neuron_model`` constants, and :mod:`training.config` vocabulary. Holds no
schema, cost, or session-building logic (those live in sibling modules), so
every other ``training`` module can import these types without a cycle.

``TargetPack`` carries three cross-cutting readout controls:

* ``readout_kind`` -- ``'ca'`` (default) or ``'v'`` delta-Vm training (#2).
* ``cost_time_ix`` -- optional sparse post-onset step indices; the target
  ``data`` stays full length and the subsample is gathered at cost time (#4).
* ``always_waveform_mse`` -- spot targets always need a waveform MSE readout;
  moving-bar targets only when a cost window was built. Encoded here so
  ``neuron_model.readout`` needs no paradigm knowledge.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from config import SIM_DTYPE_DEFAULT
from neuron_model import IH_DIR_REVERSE_CELLS

from training.config import SPOT_TARGETS


def active_device():
    """Pick CUDA or CPU from current runtime (not frozen at import)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass(frozen=True)
class TargetPack:
    """One training target: stimulus + readout indices + target traces.

    Spot ``signal`` / ``data`` time dims: :mod:`config`. Moving bar uses
    ``COST_WINDOW`` and per-target ``maxtime``.
    """

    name: str
    signal: torch.Tensor  # (B, T, N)
    data: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_weight: torch.Tensor  # (n_cost,)
    readout_batch: torch.Tensor  # (n_cost,)
    readout_unit: torch.Tensor  # (n_cost,)
    cost_t0: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed targets
    cost_radius: Optional[torch.Tensor] = None  # (n_cost,) Euclidean radius for network spot
    readout_stim_u: Optional[torch.Tensor] = None  # (n_cost,) stim anchor u per spot cost row
    readout_stim_v: Optional[torch.Tensor] = None  # (n_cost,) stim anchor v per spot cost row
    cost_extent: Optional[int] = None  # network hex-disc radius for cost readouts
    cost_pd_nd: Optional[torch.Tensor] = None  # (n_cost,) long; 0=PD, 1=ND (moving_bar)
    dsi_pos_rows: Optional[torch.Tensor] = None  # flat cost-row idx (right|up)
    dsi_neg_rows: Optional[torch.Tensor] = None  # flat cost-row idx (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_target: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_weight: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None  # scalar
    readout_kind: str = "ca"  # 'ca' or 'v' delta-Vm training target/readout
    cost_time_ix: Optional[torch.Tensor] = None  # (n_sample,) sparse post-onset step idx
    always_waveform_mse: bool = True  # spot: True; moving bar: False


@dataclass(frozen=True)
class ModelBackend:
    """Connectivity + leak/Ih tensors for one simulation graph."""

    conn: object
    e_leak: torch.Tensor
    ih_dir: torch.Tensor
    n_types: int
    n_cols: int
    network: Optional[object] = None
    depol_cells: Tuple[int, ...] = ()
    ih_reverse_cells: Tuple[int, ...] = IH_DIR_REVERSE_CELLS

    @property
    def n_units(self) -> int:
        return self.conn.n_units


@dataclass(frozen=True)
class FusedConductanceForward:
    """Conductance packs with identical signal (T, N); one ``run_full`` per group."""

    subpacks: Tuple[TargetPack, ...]
    batch_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one training / plotting run."""

    backend: ModelBackend
    model: str
    schema: tuple
    targets: Dict[str, TargetPack]
    target_list: Tuple[str, ...]
    cost_weights: Dict[str, float]
    sequential: bool
    device: str
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT
    train_opts: Optional[dict] = None
    cost_subpacks: Dict[str, TargetPack] = field(default_factory=dict)
    fused_conductance: Tuple[FusedConductanceForward, ...] = ()

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=tuple(schema))

    @property
    def primary_pack(self) -> TargetPack:
        return self.targets[self.target_list[0]]

    @property
    def maxtime(self) -> int:
        sig = self.primary_pack.signal
        return int(sig.shape[1] if sig.dim() == 3 else sig.shape[0])

    def pack_signal(self, pack: Optional[TargetPack] = None) -> torch.Tensor:
        pack = pack or self.primary_pack
        sig = pack.signal
        if pack.name in SPOT_TARGETS and sig.dim() == 3 and int(sig.shape[0]) == 1:
            sig = sig.squeeze(0)
        return sig

    def pack_for(self, name: str) -> TargetPack:
        if name not in self.targets:
            raise KeyError(f"target pack {name!r} not in session")
        return self.targets[name]


@dataclass(frozen=True)
class TrainingResult:
    """Output of :func:`training.cost.do_many_runs` (in memory; persistence is ``train``)."""

    all_params: np.ndarray   # (nofruns, n_params)
    final_costs: np.ndarray  # (nofruns,) weighted total
    best_i: int
    cost_curve: np.ndarray   # per-step weighted total for ``best_i``
    cost_curves_by_target: Dict[str, np.ndarray] = field(default_factory=dict)
    final_costs_by_target: Dict[str, np.ndarray] = field(default_factory=dict)
