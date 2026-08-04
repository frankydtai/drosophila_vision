# -*- coding: utf-8 -*-
"""Runtime dataclasses for one training / plotting run.

Leaf of the ``training`` package: depends only on :mod:`config`, the
``neuron`` constants, and :mod:`training.config` vocabulary. Holds no
schema, cost, or session-building logic (those live in sibling modules), so
every other ``training`` module can import these types without a cycle.

``ReadoutPack`` carries cross-cutting readout / drive controls:

* ``cost_time_ix`` -- optional sparse post-onset t indices; ``gt``
  stays full length and the subsample is gathered at cost time (#4).
* ``waveform_mse`` -- whether this pack needs a waveform MSE readout
  (spot: always True; moving-bar: True when a cost window was built).
  Encoded here so ``neuron.readout`` needs no paradigm knowledge.

Model traces are absolute ``v``; cost compares ``v`` to
``a_gt * gt + bias_gt``. ImpR / RecF spot gt are used as-is before affine.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from neuron import IH_DIR_REVERSE_CELLS
from neuron.params import e_ih_off, membrane_dt_over_c

from training.config import SPOT_TASKS
from param_defaults import FP


def active_device():
    """Pick CUDA or CPU from current runtime (not frozen at import)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


_FP_DTYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}


def sim_dtype_from_fp(fp: int) -> torch.dtype:
    """Map ``--fp`` / opts ``fp`` (16|32|64) to simulation tensor dtype."""
    try:
        return _FP_DTYPE[int(fp)]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"fp must be 16, 32, or 64; got {fp!r}") from e


SIM_DTYPE = sim_dtype_from_fp(FP)


@dataclass(frozen=True)
class ReadoutPack:
    """One training pack: task drive + readout indices + gt traces.

    Spot ``i_sti`` / ``gt`` time dims follow ``neuron`` / task
    timing. Moving bar uses ``COST_WINDOW`` and per-task ``n_t``.
    """

    name: str
    i_sti: torch.Tensor  # (B, T, N)
    gt: torch.Tensor  # (n_cost, T')
    power: torch.Tensor  # scalar
    cost_weight: torch.Tensor  # (n_cost,)
    readout_batch: torch.Tensor  # (n_cost,)
    readout_node: torch.Tensor  # (n_cost,)
    cost_t0: Optional[torch.Tensor] = None  # (n_cost,) absolute step for windowed readouts
    cost_radius: Optional[torch.Tensor] = None  # (n_cost,) Euclidean radius for network spot
    readout_stim_u: Optional[torch.Tensor] = None  # (n_cost,) stim anchor u per spot cost row
    readout_stim_v: Optional[torch.Tensor] = None  # (n_cost,) stim anchor v per spot cost row
    cost_extent: Optional[int] = None  # network hex-disc radius for cost readouts
    cost_pd_nd: Optional[torch.Tensor] = None  # (n_cost,) long; 0=PD, 1=ND (moving_bar)
    dsi_pos_rows: Optional[torch.Tensor] = None  # flat cost-row idx (right|up)
    dsi_neg_rows: Optional[torch.Tensor] = None  # flat cost-row idx (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None  # (n_dsi+1,) CSR
    dsi_gt: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_weight: Optional[torch.Tensor] = None  # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None  # scalar
    cost_time_ix: Optional[torch.Tensor] = None  # (n_sample,) sparse post-onset t idx
    waveform_mse: bool = True  # spot: True; moving bar: set at build
    t_onset: Optional[int] = None  # explicit onset; spot when ms_post extends i_sti past gt


@dataclass(frozen=True)
class ModelBackend:
    """Connectivity + leak/Ih tensors for one simulation graph."""

    conn: object
    e_leak: torch.Tensor
    ih_dir: torch.Tensor
    n_cells: int
    n_hexes: int
    network: Optional[object] = None
    depol_cells: Tuple[int, ...] = ()
    ih_reverse_cells: Tuple[int, ...] = IH_DIR_REVERSE_CELLS

    @property
    def n_nodes(self) -> int:
        return self.conn.n_nodes


@dataclass(frozen=True)
class FusedForward:
    """Packs with matching i_sti shape / onset; one ``forward_full`` per fuse."""

    subpacks: Tuple[ReadoutPack, ...]
    batch_offsets: Tuple[int, ...]


@dataclass(frozen=True)
class TrainSession:
    """Immutable runtime context for one training / plotting run.

    Membrane / synapse scalars are flat fields (injected from
    ``param_defaults`` at session open). ``delta_ms`` comes only from
    stimulus opts — never a separate Physics bag.
    """

    backend: ModelBackend
    model: str
    schema: tuple
    readouts: Dict[str, ReadoutPack]
    tasks: Tuple[str, ...]
    cost_weights: Dict[str, float]
    sequential: bool
    device: str
    delta_ms: float
    capac: float
    g_leak: float
    g_in: float
    E_exc: float
    E_inh: float
    E_Ih: float
    E_LEAK_REST: float
    E_LEAK_DEPOL: float
    Ih_gain: float
    Ca_tau: float
    DATA_AMP: float
    STATE_CLAMP: float
    syn_scale_exc: float
    syn_scale_inh: float
    euler: str
    pre_steady: str = "solve"
    pre_steady_iters: int = 60
    pre_steady_damp: float = 1.0
    sim_dtype: torch.dtype = SIM_DTYPE
    train_opts: Optional[dict] = None
    cost_subpacks: Dict[str, ReadoutPack] = field(default_factory=dict)
    fused_forward: Tuple[FusedForward, ...] = ()

    def with_schema(self, schema) -> "TrainSession":
        return replace(self, schema=tuple(schema))

    @property
    def dt_over_c(self) -> float:
        return membrane_dt_over_c(self.capac, self.delta_ms)

    @property
    def E_IH_OFF(self) -> float:
        return e_ih_off(self.E_LEAK_REST, self.E_Ih)

    @property
    def primary_readout(self) -> ReadoutPack:
        return self.readouts[self.tasks[0]]

    @property
    def n_t(self) -> int:
        i_sti = self.primary_readout.i_sti
        return int(i_sti.shape[1] if i_sti.dim() == 3 else i_sti.shape[0])

    def pack_i_sti(self, pack: Optional[ReadoutPack] = None) -> torch.Tensor:
        pack = pack or self.primary_readout
        i_sti = pack.i_sti
        if pack.name in SPOT_TASKS and i_sti.dim() == 3 and int(i_sti.shape[0]) == 1:
            i_sti = i_sti.squeeze(0)
        return i_sti

    def pack_for(self, name: str) -> ReadoutPack:
        if name not in self.readouts:
            raise KeyError(f"readout pack {name!r} not in session")
        return self.readouts[name]


@dataclass(frozen=True)
class TrainingResult:
    """Output of :func:`training.cost.do_many_runs` (in memory; persistence is ``train``)."""

    all_params: np.ndarray   # (nofruns, n_params)
    final_costs: np.ndarray  # (nofruns,) weighted total
    cost_curve: np.ndarray   # per-step weighted total for ``argmin(final_costs)``
    cost_curves_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
    final_costs_by_part: Dict[str, np.ndarray] = field(default_factory=dict)
