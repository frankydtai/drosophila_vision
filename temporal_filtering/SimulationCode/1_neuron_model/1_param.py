# -*- coding: utf-8 -*-
"""Shared neuron-model constants and trainable parameter numeric defaults.

Biophysics / clock constants (used by borst and hp_lp) live here so
network / stimulus / training import them from ``neuron_model.param`` without
a top-level config bag.

Schema segment *structure* (name, kind, count, indi/shared/fixed partitions)
stays in ``neuron_model.schema``; this module is the single place to edit box
bounds, initialisation numbers, and default partition tokens.

``fixed_val`` is used for units in the fixed partition when present (else ``init``).

Optional ``scale`` (default ``linear``): ``log`` stores ``z = log(physical)`` in the
optimizer; ``inv`` stores ``z = 1/physical``. In both cases
``lo``/``hi``/``init``/``fixed_val``/``carry`` remain physical units.
``jit`` for ``scale='log'`` is in natural-log units; for ``scale='inv'`` it is in
1/physical units.
"""
from __future__ import annotations

from typing import Tuple

# Simulation sampling interval (ms per discrete step).
DELTAT_MS = 10.0


def ms_to_steps(ms: float, *, deltat_ms: float = DELTAT_MS) -> int:
    """Convert milliseconds to simulation steps (rounded)."""
    return int(round(float(ms) / float(deltat_ms)))


# simulation step
deltat = DELTAT_MS  # [ms]

# borst membrane
g_leak = 1.0  # nS
E_exc = +10.0  # mV
E_inh = -70.0  # mV
capac = +40.0  # pF → 50 ms τ_m when g_leak = 1 nS
cdt = capac / deltat

Ca_tau = 50.0  # ms Ca readout
DATA_AMP = 20.0  # pA scale on ImpR target traces (fit cells)

E_LEAK_REST = -50.0
E_LEAK_DEPOL = -20.0
LEAK_DEPOL_TYPES = ['L1', 'L2', 'L3']  # [] -> all E_LEAK_REST

exc_synweight = 0.001
inh_synweight = 0.001
# Fixed edge scales passed to ScatterConn as exc_scale / inh_scale (not trainable).
# Trainable synaptic α lives in P["syn_strength"] / P["edge_weight"] (schema +
# ScatterConn._edge_alpha); SYN_MODES / synaptic_scale live in neuron_model.schema.

# Ih
E_Ih = +50.0  # mV, ON-channel reversal
E_IH_OFF = -150.0  # OFF-channel reversal (2*E_LEAK_REST - E_Ih)
Ih_gain = 1.0

IH_OFF_MODES = ("on", "off", "mirrored")
IH_OFF_DEFAULT = "on"
IH_OFF_SCALAR_SEGMENTS = frozenset({"Ih_midv_off", "Ih_slope_off", "tau_midv_off"})
IH_OFF_GMAX_SEGMENT = "Ih_gmax_off"
IH_DIR_REVERSE_CELLS: Tuple[int, ...] = ()

# hp_lp / explicit-Euler state clamp
STATE_CLAMP = 1.0e6

# registered model names
KNOWN_MODELS = ("borst", "hp_lp")

# Shared gain box (in_gain / out_gain / out_scale upper end; syn_strength hi).
GAIN_LO = 0.1
GAIN_HI = 100.0

# Lamina types with trainable Ih_gmax / Ih_gmax_off / hp_gain (L3 fixed).
DEFAULT_IH_GMAX_INDI_NAMES = ('L1', 'L2', 'L4', 'L5')

P = {
    # --- borst + hp_lp shared gains / readout ---
    "in_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1, jit=0.2),
    "out_gain": dict(lo=GAIN_LO, hi=GAIN_HI, init=1, jit=0.2),
    "out_scale": dict(lo=GAIN_LO, hi=GAIN_HI, init=1, jit=0.1),
    # --- borst type→type α (network ScatterConn; --syn-mode type_pair) ---
    "syn_strength": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    # --- per-edge magnitude (--syn-mode per_edge; sign fixed in base_w) ---
    "edge_weight": dict(lo=GAIN_LO, hi=GAIN_HI, init=1.0, jit=0.1),
    # --- borst release threshold (mV); below → no transmission ---
    "v_th": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=0.0, fixed_val=-50.0),
    # --- borst Ih ---
    "Ih_gmax": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_gmax_off": dict(lo=0.0, hi=100.0, init=50.0, jit=10.0, fixed_val=0.0),
    "Ih_midv": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    "Ih_midv_off": dict(lo=-70.0, hi=-30.0, init=-50.0, jit=5.0),
    "Ih_slope_off": dict(lo=-0.40, hi=-0.20, init=-0.25, jit=0.02),
    "tau_midv_off": dict(lo=-70.0, hi=-40.0, init=-50.0, jit=5.0),
    # --- hp_lp ---
    "tau_lp": dict(lo=DELTAT_MS, hi=100.0, init=50.0, jit=10.0),
    "bias": dict(lo=-20.0, hi=20.0, init=0.0, jit=0.1),
    "tau_hp": dict(lo=100.0, hi=10000.0, init=200.0, jit=0.0001, fixed_val=10000.0),
    "hp_gain": dict(lo=0.0, hi=5.0, init=1.0, jit=0.1, fixed_val=1.0),
}
