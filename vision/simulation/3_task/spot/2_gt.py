# -*- coding: utf-8 -*-
"""Spot paradigm GT: RecF x ImpR gt traces and cost-radius readout.

Merges the old ``Medulla_Library`` RecF/ImpR reader (with its internal
bandpass/lowpass ImpR shaping -- a gt-only spot-drive path, not the unused
Ca filter in ``neuron.filter_ca``) and the old network spot-gt section
(gt assembly + Euclidean cost radii).

New features handled here:
- ``ms_spot`` (#1): the PR drive comes from
  :func:`task.spot.input.spot_input_waveform`, shared by the network ``i_sti``
  and the ImpR gt.

ImpR / RecF traces are the raw training gt; cost compares absolute model
``v`` to ``a_gt * gt + bias_gt``. Sparse cost time points (#4) and the
``ReadoutPack`` wrapping live in the ``training`` layer, which reads the
:class:`SpotGt` returned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from network import path  # noqa: F401 -- FAFBv783 on sys.path
from import_bootstrap import parse_comma_list
from network.construction import (
    hex2gt,
    hex_in_cost_extent,
    present_gt_cells,
    normalize_gt_cells,
    node_cell_names,
)
from neuron.params import ms_to_t
from task.spot.input import (
    SpotBatch,
    euclid_hex_dist,
    members_by_euclid_radius,
    normalize_spot_timing,
    spot_extent_folds_r2_into_r1,
    spot_input_waveform,
    spot_stimulus_batches,
    spot_from_opts,
)

# ImpR / RecF gt row order (13 gt cells).
GT_CELLS: Tuple[str, ...] = (
    "L1", "L2", "L3", "L4", "L5", "Mi1", "Tm3", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
)


def expand_gt_cells(names: Sequence[str]) -> Tuple[str, ...]:
    """Validate ``--gt`` spot cell tokens against ``GT_CELLS`` (final keep-set)."""
    if not names:
        raise ValueError("gt_cells must not be empty")
    out: list = []
    seen: set = set()
    for raw in names:
        key = str(raw).strip()
        if key not in GT_CELLS:
            valid = ", ".join(GT_CELLS)
            raise ValueError(f"unknown gt cell {key!r} (expected {valid})")
        if key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)

# Spot paradigm polarities (distinct from the task NAMES in training.config).
SPOT_POLARITIES = frozenset({"bright", "dark"})
_SPOT_BASELINE_KEY = "i_baseline_spot"
_SPOT_I_KEY = {"bright": "i_bright_spot", "dark": "i_dark_spot"}

# RecF sample grid: center at sample 22; one integer radius step = 5 samples.
# Profile cube axis is Euclidean radius (0 .. RF_N_RADII-1), not a mirrored bin.
_RF_CENTER_SAMPLE = 22
_RF_SAMPLES_PER_COL = 5
_RF_NSAMPLES = 45
RF_CENTER_RADIUS = 0
RF_N_RADII = 5
RF_RADIUS_DEG = _RF_SAMPLES_PER_COL  # degrees per integer radius on RF plots
# Gt-only ImpR shaping helpers (not the unused Ca filter). Inlined from the
# old blindschleiche_py3 module so spot/gt owns this path alone.


def _gauss1d(fwhm, rfsize):
    myrange = rfsize / 2
    sigma = fwhm / (2.0 * np.sqrt(2 * np.log(2)))
    x = np.arange(-myrange, (myrange + 1), 1) * 1.0
    z = np.exp(-x ** 2 / (2 * (sigma ** 2)))
    return z / np.sum(z)


def _lowpass(x, tau_ms, *, delta_ms: float):
    """Euler low-pass; ``tau_ms`` is the time constant in milliseconds."""
    x = x.transpose(np.roll(np.arange(x.ndim), 1))
    n = x.shape[0]
    result = np.zeros_like(x)
    tau_ms = float(tau_ms)
    dt = float(delta_ms)
    if dt <= 0:
        raise ValueError(f"delta_ms must be > 0, got {dt}")
    if tau_ms < dt:
        result = x
    else:
        alpha = dt / tau_ms
        result[0] = x[0]
        for i in range(0, n - 1):
            result[i + 1] = alpha * (x[i] - result[i]) + result[i]
    return result.transpose(np.roll(np.arange(result.ndim), -1))


def _bandpass(x, hp_tau_ms, lp_tau_ms, *, delta_ms: float):
    result = _lowpass(x, lp_tau_ms, delta_ms=delta_ms)
    if hp_tau_ms != 0:
        result = result - _lowpass(result, hp_tau_ms, delta_ms=delta_ms)
    return result


def normalize_gt(x):
    x = x - x[0]
    mymax = np.nanmax(x)
    mymin = np.nanmin(x)
    if mymax == mymin:
        return x * 0.0
    return x / max(abs(mymax), abs(mymin))


def _shift_right(y, k: int):
    """Delay ``y`` by ``k`` samples (leading zeros; trailing samples dropped)."""
    y = np.asarray(y)
    k = int(k)
    if k <= 0:
        return y
    out = np.zeros_like(y)
    out[k:] = y[:-k]
    return out


# ImpR onset delay (samples / t-index); same for all gt cells.
_IMPR_SHIFT = 5


def read_RecF_ImpR(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Return ``(RecF_gt, ImpR_gt)`` for the 13 gt cells.

    Shapes: ``RecF_gt`` ``(13, 45)``; ``ImpR_gt`` ``(13, n_t)``. The
    drive is :func:`task.spot.input.spot_input_waveform` (step or finite spot).
    ImpR filter taus are in ms (scaled by ``delta_ms``); delay is in samples.
    """
    if t_onset is None or n_t is None:
        raise ValueError("read_RecF_ImpR requires t_onset and n_t")
    t_onset = int(t_onset)
    n_t = int(n_t)
    delta_ms = float(delta_ms)
    if delta_ms <= 0:
        raise ValueError(f"delta_ms must be > 0, got {delta_ms}")
    n_cells = len(GT_CELLS)

    RF_center_width = np.array([6, 7, 6, 8, 7, 6, 12, 6, 6, 8, 8, 11, 7])
    RF_surrnd_width = np.array([41, 29, 15, 33, 31, 29, 7, 16, 24, 27, 31, 35, 24])
    RF_surrnd_weight = np.array(
        [0.012, 0.013, 0.19, 0.046, 0.035, 0.022, 0.000, 0.132, 0.063, 0.040, 0.035, 0.054, 0.046]
    ) * 5.0
    RF_sign = np.array([-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

    RecF_gt = np.zeros((n_cells, 45))
    for i in range(n_cells):
        center = _gauss1d(RF_center_width[i], 44)
        surrnd = _gauss1d(RF_surrnd_width[i], 44)
        RecF_gt[i] = normalize_gt(
            (center - RF_surrnd_weight[i] * surrnd) * RF_sign[i]
        )

    # ImpR HP / LP time constants (ms).
    IR_hp_ms = np.array(
        [391.0, 288.0, 0.0, 381.0, 127.0, 318.0, 260.0, 0.0, 0.0, 296.0, 153.0, 249.0, 0.0]
    )
    IR_lp_ms = np.array(
        [38.0, 58.0, 54.0, 23.0, 42.0, 54.0, 27.0, 38.0, 77.0, 44.0, 14.0, 24.0, 107.0]
    )

    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    u = u / np.max(u)

    ImpR_gt = np.zeros((n_cells, n_t))
    for i in range(n_cells):
        ImpR_gt[i] = _shift_right(
            normalize_gt(_bandpass(u, IR_hp_ms[i], IR_lp_ms[i], delta_ms=delta_ms)),
            _IMPR_SHIFT,
        )

    return RecF_gt, ImpR_gt


def read_RecF_gt(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Spatial x temporal spot cube ``(n_cells, RF_N_RADII, n_t)``; axis = radius."""
    RecF_gt, ImpR_gt = read_RecF_ImpR(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
    )
    mt = ImpR_gt.shape[1]
    n_cells = len(GT_CELLS)
    gt = np.zeros((n_cells, RF_N_RADII, mt))
    for i in range(n_cells):
        for radius in range(RF_N_RADII):
            sample = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
            gt[i, radius] = RecF_gt[i, sample] * ImpR_gt[i]
    return gt


def read_RecF_gt_dark(*, t_onset=None, n_t=None, ms_spot=None, delta_ms: float):
    """Dark spot spatial x temporal cube: negated bright ``read_RecF_gt()``."""
    return -read_RecF_gt(
        t_onset=t_onset, n_t=n_t, ms_spot=ms_spot, delta_ms=delta_ms,
    )


# -- Cost-radius weights ------------------------------------------------------


def normalize_spot_cost_radius_key(key, *, aliases: Dict[str, float]) -> float:
    if isinstance(key, (int, float)):
        return round(float(key), 6)
    text = str(key).strip().lower()
    if text in aliases:
        return round(float(aliases[text]), 6)
    return round(float(text), 6)


def parse_spot_cost_radius_weight_value(text: str) -> float:
    tok = str(text).strip()
    if "/" in tok:
        num, den = tok.split("/", 1)
        return float(num.strip()) / float(den.strip())
    return float(tok)


def expand_spot_cost_r_w_dict(
    kv: Optional[dict] = None,
    *,
    stimulus_opts: Optional[dict] = None,
    aliases: Dict[str, float],
) -> Optional[Dict[float, float]]:
    if stimulus_opts is not None:
        kv = (stimulus_opts or {}).get("spot_cost_radius_weight")
    if not kv:
        return None
    return {
        normalize_spot_cost_radius_key(k, aliases=aliases): parse_spot_cost_radius_weight_value(v)
        for k, v in kv.items()
    }


def expand_cost_ms_dict(
    *,
    stimulus_opts: Optional[dict] = None,
    aliases: Dict[str, float],
) -> Dict[float, Tuple[float, ...]]:
    """Radius → explicit post-onset ms; empty when unset."""
    kv = (stimulus_opts or {}).get("cost_ms") if stimulus_opts is not None else None
    if not kv:
        return {}
    out: Dict[float, Tuple[float, ...]] = {}
    for k, v in kv.items():
        r = normalize_spot_cost_radius_key(k, aliases=aliases)
        vals = parse_comma_list(v) if isinstance(v, str) else list(v)
        if not vals:
            raise ValueError(f"cost_ms[{k!r}] must list at least one ms")
        out[r] = tuple(float(x) for x in vals)
    return out


def parse_cost_ms_tokens(
    tokens: Optional[Sequence[str]],
    *,
    aliases: Dict[str, float],
) -> Optional[Dict[float, Tuple[float, ...]]]:
    """Parse ``--cost-ms``: ``none``/``off`` → ``{}``; else ``R=MS,...``. Omit → ``None``."""
    if tokens is None:
        return None
    if len(tokens) == 1 and str(tokens[0]).strip().lower() in ("none", "off"):
        return {}
    out: Dict[float, Tuple[float, ...]] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"expected R=MS,... or none|off, got {tok!r}")
        key, val = tok.split("=", 1)
        vals = parse_comma_list(val)
        if not vals:
            raise ValueError(f"--cost-ms {key}=... must list at least one ms")
        out[normalize_spot_cost_radius_key(key, aliases=aliases)] = tuple(
            float(x) for x in vals
        )
    return out


def default_spot_cost_radius_weight(
    spot_extent: float,
    *,
    weights: Dict[float, float],
    weights_extent1: Dict[float, float],
) -> Dict[float, float]:
    """Cost-radius weights for ``spot_extent`` (extent-1 folds r=2 into r=1)."""
    if spot_extent_folds_r2_into_r1(spot_extent):
        return dict(weights_extent1)
    return dict(weights)


def parse_spot_cost_r_w_tokens(
    tokens: Optional[Sequence[str]],
    *,
    default_weights: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    aliases: Dict[str, float],
) -> Optional[Dict[float, float]]:
    """Parse ``--spot-cost-r-w`` space-separated ``R`` / ``R=W`` tokens."""
    if not tokens:
        return None
    bare: list[float] = []
    explicit: Dict[float, float] = {}
    for tok in tokens:
        if "=" in tok:
            key, val = tok.split("=", 1)
            explicit[normalize_spot_cost_radius_key(key, aliases=aliases)] = (
                parse_spot_cost_radius_weight_value(val)
            )
        else:
            bare.append(normalize_spot_cost_radius_key(tok, aliases=aliases))
    if bare:
        weights = {round(float(r), 6): 0.0 for r in spot_cost_radii}
        for r in bare:
            weights[r] = 1.0
    else:
        weights = dict(default_weights)
    weights.update(explicit)
    return weights


def resolve_spot_cost_radii(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    *,
    default_weights: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    stimulus_opts: Optional[dict] = None,
    aliases: Optional[Dict[str, float]] = None,
) -> Tuple[float, ...]:
    if stimulus_opts is not None:
        if aliases is None:
            raise ValueError("resolve_spot_cost_radii with stimulus_opts requires aliases")
        spot_cost_radius_weight = expand_spot_cost_r_w_dict(
            stimulus_opts=stimulus_opts, aliases=aliases,
        )
    weights = (
        dict(default_weights)
        if spot_cost_radius_weight is None
        else spot_cost_radius_weight
    )
    return tuple(
        radius for radius in spot_cost_radii
        if float(weights.get(round(radius, 6), 0.0)) != 0.0
    )


def spot_sti_radius_gate(
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    *,
    default_weights: Dict[float, float],
    spot_sti_radii: Tuple[float, ...],
) -> Tuple[float, ...]:
    """Per ``spot_sti_radii`` slot: ``1`` if cost-radius weight ≠ 0 else ``0``.

    Forward multiplies ``a_sti_radius`` by this gate (indi or fixed).
    """
    weights = (
        dict(default_weights)
        if spot_cost_radius_weight is None
        else spot_cost_radius_weight
    )
    return tuple(
        0.0 if float(weights.get(round(float(r), 6), 0.0)) == 0.0 else 1.0
        for r in spot_sti_radii
    )


def spot_cost_node_weight(
    radius: float,
    spot_cost_radius_weight: Optional[Dict[float, float]],
    *,
    default_weights: Dict[float, float],
) -> float:
    weights = (
        dict(default_weights)
        if spot_cost_radius_weight is None
        else spot_cost_radius_weight
    )
    return float(weights.get(round(radius, 6), 0.0))


# -- RecF sampling / superposed gt ----------------------------------------


def _recf_at(recf_row: np.ndarray, radius: float) -> float:
    idx = _RF_CENTER_SAMPLE + _RF_SAMPLES_PER_COL * radius
    idx = min(max(idx, 0.0), _RF_NSAMPLES - 1)
    return float(np.interp(idx, np.arange(_RF_NSAMPLES), recf_row))


def _spot_readout_amp(recf_row: np.ndarray, radius: float, spot_extent: float) -> float:
    r = round(float(radius), 6)
    if spot_extent_folds_r2_into_r1(spot_extent):
        if r == 1.0:
            return _recf_at(recf_row, 1.0) + _recf_at(recf_row, 2.0)
        if r == 2.0:
            return 0.0
    return _recf_at(recf_row, r)


def _spot_superposed_trace(
    recf_row: np.ndarray,
    stim_uv: Sequence[Tuple[int, int]],
    mu: int,
    mv: int,
    spot_extent: float,
    impr_row: np.ndarray,
    resp: slice,
    data_amp: float,
    *,
    polarity: str,
) -> np.ndarray:
    amp = 0.0
    for su, sv in stim_uv:
        dist = round(euclid_hex_dist(int(mu) - int(su), int(mv) - int(sv)), 6)
        amp += _spot_readout_amp(recf_row, dist, spot_extent)
    trace = amp * impr_row[resp] * data_amp
    if polarity == "dark":
        trace = -trace
    return trace


def spot_cost_hexes(
    batches: Sequence[SpotBatch],
    cost_radii,
    cost_extent,
) -> List[Tuple[int, int, int, float, int, int]]:
    """Cost readouts: ``(batch, mu, mv, radius_key, su, sv)`` per stim radius."""
    by_radius = members_by_euclid_radius(cost_radii)
    cols: List[Tuple[int, int, int, float, int, int]] = []
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            for radius_key, members in by_radius.items():
                for du, dv in members:
                    mu, mv = su + du, sv + dv
                    if not hex_in_cost_extent(mu, mv, cost_extent):
                        continue
                    cols.append((
                        b, int(mu), int(mv), float(radius_key), int(su), int(sv),
                    ))
    return cols


def spot_n_cost_hexes(cost_hexes):
    if not cost_hexes:
        return 0
    counts: Dict[int, int] = {}
    for b, _mu, _mv, _radius, _su, _sv in cost_hexes:
        counts[b] = counts.get(b, 0) + 1
    vals = set(counts.values())
    if len(vals) == 1:
        return next(iter(vals))
    return {b: counts[b] for b in sorted(counts)}


def _as_np(arr) -> np.ndarray:
    return arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)


def build_spot_cost_readout(C, batches, cost_radii, cost_extent):
    u = _as_np(C.u)
    v = _as_np(C.v)
    type_all = _as_np(C.node_cell)
    batch_idx, node_idx, radius, type_idx, stim_u, stim_v = [], [], [], [], [], []
    for b, mu, mv, cell_radius, su, sv in spot_cost_hexes(
        batches, cost_radii, cost_extent,
    ):
        on_col = (u == mu) & (v == mv)
        for uid in np.where(on_col)[0]:
            batch_idx.append(b)
            node_idx.append(int(uid))
            radius.append(cell_radius)
            type_idx.append(int(type_all[uid]))
            stim_u.append(int(su))
            stim_v.append(int(sv))
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(node_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
        np.asarray(stim_u, dtype=np.int64),
        np.asarray(stim_v, dtype=np.int64),
    )


def build_spot_center_readout(C, batches, cost_radii, cost_extent):
    """Cost-node readout plus ``center_row`` mask for stim-on hex (radius 0)."""
    batch_idx, node_idx, radius, type_idx, stim_u, stim_v = build_spot_cost_readout(
        C, batches, cost_radii, cost_extent,
    )
    stim_u = np.asarray(stim_u, dtype=np.int64)
    stim_v = np.asarray(stim_v, dtype=np.int64)
    u_all = _as_np(C.u)
    v_all = _as_np(C.v)
    du = np.asarray(u_all[node_idx] - stim_u, dtype=np.int64)
    dv = np.asarray(v_all[node_idx] - stim_v, dtype=np.int64)
    center_row = (du == 0) & (dv == 0)
    return (
        np.asarray(batch_idx, dtype=np.int64),
        np.asarray(node_idx, dtype=np.int64),
        np.asarray(radius, dtype=np.float64),
        np.asarray(type_idx, dtype=np.int64),
        stim_u,
        stim_v,
        du,
        dv,
        center_row,
    )


@dataclass
class SpotGt:
    i_sti: torch.Tensor          # (B, T, N)
    gt: torch.Tensor            # (n_cost, T')
    power: torch.Tensor           # scalar
    cost_weight: torch.Tensor     # (n_cost,)
    cost_radius: torch.Tensor     # (n_cost,)
    readout_batch: torch.Tensor   # (n_cost,) long
    readout_node: torch.Tensor    # (n_cost,) long
    readout_stim_u: torch.Tensor  # (n_cost,) long
    readout_stim_v: torch.Tensor  # (n_cost,) long
    n_batch: int
    info: dict


def build_spot_gt(
    C,
    *,
    spot_extent: float,
    multi_spot: bool,
    fully_inside: bool,
    shift_extent: int,
    n_t: int,
    t_onset: int,
    i_baseline_spot: float,
    i_bright_spot: float,
    i_dark_spot: float,
    polarity: str,
    data_amp: float,
    delta_ms: float,
    default_cost_weights: Dict[float, float],
    spot_cost_radii: Tuple[float, ...],
    device: Optional[str] = None,
    cost_extent: Optional[int] = None,
    spot_cost_radius_weight: Optional[Dict[float, float]] = None,
    sim_dtype: torch.dtype,
    ms_spot: Optional[float] = None,
    ms_response: Optional[float] = None,
    gt_cells: Optional[Sequence[str]] = None,
) -> SpotGt:
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"polarity must be 'bright' or 'dark', got {polarity!r}")
    i_baseline = float(i_baseline_spot)
    i_spot = float(i_bright_spot if polarity == "bright" else i_dark_spot)
    device = device or C.device
    if ms_response is None:
        raise ValueError("build_spot_gt requires ms_response")
    n_t_gt = int(t_onset) + ms_to_t(float(ms_response), delta_ms=float(delta_ms)) + 1
    if n_t_gt > int(n_t):
        raise ValueError(
            f"spot gt n_t={n_t_gt} exceeds forward n_t={n_t} "
            f"(ms_response={ms_response:g}, t_onset={t_onset})"
        )
    recf_gt, impr_gt = read_RecF_ImpR(
        t_onset=t_onset, n_t=n_t_gt, ms_spot=ms_spot, delta_ms=delta_ms,
    )
    type_row = {str(rt): i for i, rt in enumerate(GT_CELLS)}
    if gt_cells is not None:
        bad = [str(t) for t in gt_cells if str(t) not in type_row]
        if bad:
            raise ValueError(
                f"unknown spot gt cell(s) {bad!r} "
                f"(expected subset of {list(GT_CELLS)})",
            )

    spot = spot_from_opts(
        C,
        spot_extent=spot_extent,
        shift_extent=shift_extent,
        multi_spot=multi_spot,
        fully_inside=fully_inside,
    )
    names = node_cell_names(C)
    present = present_gt_cells(
        gt_cells, GT_CELLS, C.cell_names, context="spot",
    )

    batches = spot_stimulus_batches(spot)
    n_batch = len(batches)

    # Single PR waveform source (step or finite spot) shared with the ImpR gt.
    u = spot_input_waveform(t_onset, n_t, ms_spot, delta_ms=delta_ms)
    drive = torch.as_tensor(
        i_baseline + (i_spot - i_baseline) * u, dtype=sim_dtype, device=device,
    )
    # All PR hexes hold i_baseline; stim_uv hexes then get the step/spot drive.
    pr_idx = torch.as_tensor(np.where(C.is_input)[0], dtype=torch.long, device=device)
    i_sti = torch.zeros((n_batch, n_t, C.n_nodes), dtype=sim_dtype, device=device)
    if len(pr_idx):
        i_sti[:, :, pr_idx] = float(i_baseline)
    for b, batch in enumerate(batches):
        for su, sv in batch.stim_uv:
            nodes = C.input_nodes_at(su, sv)
            if len(nodes):
                idx = torch.as_tensor(nodes, dtype=torch.long, device=device)
                i_sti[b, :, idx] = drive[:, None]

    resp = slice(t_onset, n_t_gt)  # cost window: response only (no ms_post)

    cost_radii = resolve_spot_cost_radii(
        spot_cost_radius_weight,
        default_weights=default_cost_weights,
        spot_cost_radii=spot_cost_radii,
    )
    cost_hexes = spot_cost_hexes(batches, cost_radii, cost_extent)

    cost_batch, cost_node, cost_radius_rows, cost_readout, cost_weight_rows = [], [], [], [], []
    cost_stim_u, cost_stim_v = [], []
    trace_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
    for b, mu, mv, radius, su, sv in cost_hexes:
        w = spot_cost_node_weight(
            radius, spot_cost_radius_weight, default_weights=default_cost_weights,
        )
        if w == 0.0:
            continue
        stim_uv = batches[b].stim_uv
        for rt in present:
            nodes = hex2gt(C, mu, mv, rt, names)
            if len(nodes) == 0:
                continue
            row = type_row[rt]
            cache_key = (b, mu, mv, row)
            if cache_key not in trace_cache:
                trace_cache[cache_key] = _spot_superposed_trace(
                    recf_gt[row],
                    stim_uv,
                    mu,
                    mv,
                    spot_extent,
                    impr_gt[row],
                    resp,
                    data_amp,
                    polarity=polarity,
                )
            trace = trace_cache[cache_key]
            for uidx in nodes:
                cost_batch.append(b)
                cost_node.append(int(uidx))
                cost_radius_rows.append(radius)
                cost_readout.append(trace)
                cost_weight_rows.append(w)
                cost_stim_u.append(int(su))
                cost_stim_v.append(int(sv))

    if not cost_batch:
        raise ValueError("no spot cost nodes (check cost_extent and gt cells)")

    gt = torch.tensor(np.asarray(cost_readout), dtype=sim_dtype, device=device)  # (n_cost,T')
    cost_weight = torch.tensor(np.asarray(cost_weight_rows), dtype=sim_dtype, device=device)
    cost_radius = torch.tensor(np.asarray(cost_radius_rows), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(cost_batch), dtype=torch.long, device=device)
    readout_node = torch.tensor(np.asarray(cost_node), dtype=torch.long, device=device)
    readout_stim_u = torch.tensor(np.asarray(cost_stim_u), dtype=torch.long, device=device)
    readout_stim_v = torch.tensor(np.asarray(cost_stim_v), dtype=torch.long, device=device)

    power = torch.sum(cost_weight[:, None] * gt ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)

    info = {
        "n_batch": n_batch,
        "n_cost": gt.shape[0],
        "n_cost_hexes": spot_n_cost_hexes(cost_hexes),
        "n_centers": len(spot.centers),
        "n_shifts": len(spot.shifts),
        "cost_extent": cost_extent,
        "spot_extent": float(spot_extent),
        "multi_spot": bool(multi_spot),
        "fully_inside": bool(fully_inside),
        "spot_cost_radius_weight": spot_cost_radius_weight,
        "spot_cost_radii": list(cost_radii),
        "present_gts": present,
        "i_baseline_spot": float(i_baseline),
        "i_bright_spot": float(i_bright_spot),
        "i_dark_spot": float(i_dark_spot),
        "polarity": str(polarity),
        "ms_spot": None if ms_spot is None else float(ms_spot),
        "ms_response": float(ms_response),
        "t_onset": int(t_onset),
        "n_t": int(n_t),
        "n_t_gt": int(n_t_gt),
    }
    return SpotGt(
        i_sti=i_sti,
        gt=gt,
        power=power,
        cost_weight=cost_weight,
        cost_radius=cost_radius,
        readout_batch=readout_batch,
        readout_node=readout_node,
        readout_stim_u=readout_stim_u,
        readout_stim_v=readout_stim_v,
        n_batch=n_batch,
        info=info,
    )


def make_spot_stimulus_opts(
    polarity: str,
    *,
    i_baseline_spot: float,
    i_spot: float,
    ms_pre: float,
    ms_response: float,
    delta_ms: float,
    delta_ms_pre: float,
    shift_extent: int,
    spot_extent: float,
    multi_spot: bool,
    fully_inside: bool,
    ms_spot=None,
    ms_post: float = 0.0,
    cost_interval_ms=None,
    cost_ms=None,
    gt_cells=None,
):
    """PR step/spot stimulus opts for ``spot_{polarity}``."""
    if polarity not in SPOT_POLARITIES:
        raise ValueError(f"spot polarity must be 'bright' or 'dark', got {polarity!r}")
    peak_key = _SPOT_I_KEY[polarity]
    opts = {
        _SPOT_BASELINE_KEY: float(i_baseline_spot),
        peak_key: float(i_spot),
        "ms_pre": float(ms_pre),
        "ms_response": float(ms_response),
        "ms_post": float(ms_post),
        "delta_ms": float(delta_ms),
        "delta_ms_pre": float(delta_ms_pre),
        "shift_extent": int(shift_extent),
        "spot_extent": float(spot_extent),
        "multi_spot": bool(multi_spot),
        "fully_inside": bool(fully_inside),
    }
    if ms_spot is not None:
        opts["ms_spot"] = float(ms_spot)
    if cost_interval_ms is not None:
        opts["cost_interval_ms"] = float(cost_interval_ms)
    if cost_ms is not None:
        opts["cost_ms"] = {
            str(float(k)): [float(x) for x in v] for k, v in cost_ms.items()
        }
    rs = normalize_gt_cells(gt_cells)
    if rs is not None:
        opts["gt_cells"] = rs
    return normalize_spot_timing(opts)
