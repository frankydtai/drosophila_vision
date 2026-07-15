# -*- coding: utf-8 -*-
"""Moving-bar stimulus + training target for connectome and Borst backends.

Column geometry and bar physics live in :mod:`visual_stimulus.moving_bar_stimulus`.
This module maps column-level currents to photoreceptor units, builds batched
``signal`` tensors, and assembles fig1 cost readouts via
:func:`build_moving_bar_target`. Window indexing for cost/plot traces is
:func:`moving_bar_window_t_rel` / :func:`moving_bar_window_t_rel_torch`;
plot ``t0`` grids use :func:`moving_bar_session_t0_grids` /
:func:`build_moving_bar_t0_grids`.

``build_moving_bar_signals`` returns ``signal`` with shape ``(B, T, N_units)``.
Default ``T`` is ``t_on`` + sweep + moving-bar tail, not Borst ``IMPULSE_MAXTIME``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import network_bootstrap  # noqa: F401

import Medulla_Library as ml
from Medulla_Library import I_BASELINE, I_BRIGHT, I_DARK
from column_mapper import borst_sti_columns
from connectome_io import moving_bar_cache_dir
from network.stimulus import column_in_cost_extent
from t4_t5_preference import (
    READOUT_SUBTYPES,
    active_stimuli_for_subtype,
    fig1_key_for_stimulus,
    motion_preference,
    normalize_side,
)
from training_config import (
    COST_ALIGNED_FIRST_STI_MS,
    COST_WINDOW_AFTER_MS,
    COST_WINDOW_BEFORE_MS,
    COST_WINDOW_MS,
    DELTAT_MS,
    FIG1_CI_NPZ,
    SIM_DTYPE_DEFAULT,
    T_ON,
    ms_to_steps,
)
from visual_stimulus.moving_bar_stimulus import (
    DEFAULT_BAR_EXTENT,
    GRUNTMAN_SPEED_DEG_S,
    HexColumn,
    MovingBarSpec,
    build_batched_column_current,
    column_first_stim_step,
    field_bounds,
    gruntman_moving_bar_specs,
    hex_column_from_uv,
    moving_bar_maxtime,
    moving_bar_sweep_end_step,
)

from .construction import unit_type_names

logger = logging.getLogger(__name__)

_TRACE_CACHE: Dict[str, np.ndarray] = {}
BORST_READOUT_SUBTYPES = ("T4a", "T4b", "T5a", "T5b")
PD_IDX, ND_IDX = 0, 1


@dataclass
class StiColumn(HexColumn):
    """One sti column on a connectome, with unit indices for scattering."""

    unit_idx: np.ndarray


def _sti_column_from_uv(u: int, v: int, unit_idx: np.ndarray) -> StiColumn:
    base = hex_column_from_uv(u, v)
    return StiColumn(
        u=base.u,
        v=base.v,
        x=base.x,
        y=base.y,
        x_deg=base.x_deg,
        y_deg=base.y_deg,
        hex_xy=base.hex_xy,
        unit_idx=unit_idx,
    )


@dataclass
class MovingBarStimulus:
    signal: torch.Tensor
    column_current: np.ndarray
    specs: List[MovingBarSpec]
    info: dict = field(default_factory=dict)


def sti_columns(C) -> List[StiColumn]:
    """Sti columns with photoreceptor units (one per axial ``(u, v)``)."""
    cols: Dict[Tuple[int, int], StiColumn] = {}
    u_in = C.u[C.is_input]
    v_in = C.v[C.is_input]
    for u, v in zip(u_in.tolist(), v_in.tolist()):
        key = (int(u), int(v))
        if key in cols:
            continue
        units = C.input_units_at(key[0], key[1])
        if len(units) == 0:
            continue
        cols[key] = _sti_column_from_uv(
            key[0], key[1], np.asarray(units, dtype=np.int64),
        )
    return [cols[k] for k in sorted(cols)]


def moving_bar_cost_columns(C, cost_extent=None) -> List[StiColumn]:
    """Sti columns used for moving-bar cost (optional central hex disc)."""
    cols = sti_columns(C)
    if cost_extent is None:
        return cols
    return [c for c in cols if column_in_cost_extent(c.u, c.v, cost_extent)]


def _as_int64_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return np.asarray(x.detach().cpu().numpy(), dtype=np.int64)
    return np.asarray(x, dtype=np.int64)


def network_uv_np(C) -> Tuple[np.ndarray, np.ndarray]:
    """Connectome axial ``(u, v)`` per unit as int64 numpy arrays."""
    return _as_int64_np(C.u), _as_int64_np(C.v)


def _coord_matches(val, axis_filter, tol=1e-6) -> bool:
    if axis_filter is None:
        return True
    if isinstance(axis_filter, (list, tuple)):
        return any(np.isclose(val, float(v), atol=tol) for v in axis_filter)
    return np.isclose(val, float(axis_filter), atol=tol)


def filter_sti_columns(cols, *, at_x=None, at_y=None, tol=1e-6):
    """Keep network sti columns whose hex-step ``(x, y)`` matches ``at_x`` / ``at_y``."""
    if at_x is None and at_y is None:
        return list(cols)
    out = []
    for col in cols:
        if not _coord_matches(col.x, at_x, tol=tol):
            continue
        if not _coord_matches(col.y, at_y, tol=tol):
            continue
        out.append(col)
    return out


def filter_borst_sti_columns(cols, *, at_x=None, at_y=None, tol=1e-6):
    """Keep Borst sti columns for ``--x`` (``k`` in -2..+2) / ``--y=0``."""
    if at_x is None and at_y is None:
        return list(cols)
    if at_y is not None and not _coord_matches(0.0, at_y, tol=tol):
        raise ValueError('Borst moving-bar --y must be 0 (horizontal row)')
    if at_x is None:
        return list(cols)
    if isinstance(at_x, (list, tuple)):
        ks = []
        for v in at_x:
            k = int(round(float(v)))
            if not np.isclose(float(v), k, atol=tol):
                raise ValueError(
                    f'Borst --x={v!r} must be an integer column k in -2..+2',
                )
            ks.append(k)
        out = [col for col in cols if col.k in ks]
        if not out:
            raise ValueError(f'no Borst columns match x={list(at_x)!r}')
        return out
    k = int(round(float(at_x)))
    if not np.isclose(float(at_x), k, atol=tol):
        raise ValueError(f'Borst --x={at_x!r} must be an integer column k in -2..+2')
    out = [col for col in cols if col.k == k]
    if not out:
        raise ValueError(f'no Borst column with k={k!r}')
    return out


def moving_bar_i_baseline_from_opts(train_opts) -> float:
    """``i_baseline`` from moving-bar stimulus opts on a train session."""
    opts = train_opts or {}
    for key in ("moving_bar_bright_stimulus_opts", "moving_bar_dark_stimulus_opts"):
        sub = opts.get(key) or {}
        if "i_baseline" in sub:
            return resolve_i_baseline(float(sub["i_baseline"]))
    return resolve_i_baseline(None)


def bar_specs_for_session(session, target) -> List[MovingBarSpec]:
    """Gruntman bar specs for ``target`` (bright/dark; connectome vs Borst directions)."""
    contrast = "bright" if "bright" in target else "dark"
    C = session.backend.network
    if C is not None:
        return list(gruntman_moving_bar_specs(contrasts=(contrast,)))
    return list(gruntman_moving_bar_specs(directions=("right", "left"), contrasts=(contrast,)))


@dataclass
class MovingBarSessionT0:
    t0_bn: np.ndarray
    before_steps: Dict[str, int]
    after_steps: Dict[str, int]
    side: str
    n_filter_cols: int


def moving_bar_session_t0_grids(
    session,
    specs: Sequence[MovingBarSpec],
    cost_extent,
    maxtime: int,
    *,
    at_x=None,
    at_y=None,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
) -> MovingBarSessionT0:
    """Session-level ``t0`` / horizon grids for moving-bar cost or analysis."""
    C = session.backend.network
    i_baseline = moving_bar_i_baseline_from_opts(session.train_opts)

    if C is not None:
        side = normalize_side(C.meta.get('side', 'right'))
        all_cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
        if at_x is not None or at_y is not None:
            filt_cols = filter_sti_columns(all_cols, at_x=at_x, at_y=at_y)
            if not filt_cols:
                raise SystemExit(
                    f'no sti columns match x={at_x!r} y={at_y!r} within cost_extent',
                )
        else:
            filt_cols = all_cols
        stim = build_moving_bar_signals(
            C, specs=specs, maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
            device=C.node_type.device, i_baseline=i_baseline,
        )
        uv_to_idx = {(int(col.u), int(col.v)): j for j, col in enumerate(sti_columns(C))}
        all_col_idxs = [uv_to_idx[(int(c.u), int(c.v))] for c in all_cols]
        filt_col_idxs = [uv_to_idx[(int(c.u), int(c.v))] for c in filt_cols]
        grids = build_moving_bar_t0_grids(
            stim.column_current, specs, maxtime, i_baseline,
            all_col_idxs=all_col_idxs,
            filt_col_idxs=filt_col_idxs,
            network_C=C,
            filt_network_cols=filt_cols,
        )
        n_filter_cols = len(filt_cols)
    else:
        side = "right"
        cols_all = list(borst_sti_columns())
        all_col_ids = list(range(ml.nofcols))
        if at_x is not None or at_y is not None:
            filt_cols = filter_borst_sti_columns(cols_all, at_x=at_x, at_y=at_y)
            if not filt_cols:
                raise SystemExit(
                    f'no Borst columns match x={at_x!r} y={at_y!r}',
                )
            col_ids = [col.col for col in filt_cols]
        else:
            col_ids = all_col_ids
        col_curr = build_batched_column_current(
            cols_all, specs, maxtime, t_on=t_on, deltat_ms=deltat_ms,
            i_baseline=i_baseline,
        )
        grids = build_moving_bar_t0_grids(
            col_curr, specs, maxtime, i_baseline,
            all_col_idxs=all_col_ids,
            filt_col_idxs=col_ids,
            borst_filt_col_ids=col_ids,
        )
        n_filter_cols = len(col_ids)

    return MovingBarSessionT0(
        t0_bn=grids.t0_bn,
        before_steps=grids.before_steps,
        after_steps=grids.after_steps,
        side=side,
        n_filter_cols=n_filter_cols,
    )


def _pack_readout_type_names(session, target: str) -> List[str]:
    """Unique cell-type names on ``pack.readout_unit`` (pack order)."""
    pack = session.pack_for(target)
    u = pack.readout_unit
    if torch.is_tensor(u):
        u = u.detach().cpu().numpy()
    u = np.asarray(u, dtype=np.int64)
    backend = session.backend
    if backend.network is not None:
        C = backend.network
        node_type = C.node_type[u]
        if torch.is_tensor(node_type):
            node_type = node_type.detach().cpu().numpy()
        names = list(C.type_names)
        seq = [str(names[int(t)]) for t in node_type]
    else:
        node_type = backend.conn.node_type[u]
        if torch.is_tensor(node_type):
            node_type = node_type.detach().cpu().numpy()
        seq = [str(ml.ctype[int(t)]) for t in node_type]
    seen: set = set()
    out: List[str] = []
    for name in seq:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def moving_bar_row_specs(session, target: str, side: str) -> Dict[str, List[str]]:
    """Per-readout-cell active bar spec names for ``side`` and target contrast."""
    contrast = "bright" if "bright" in target else "dark"
    return {
        st: [
            f'{d}_{c}_{w}'
            for d, c, w in active_stimuli_for_subtype(side, st)
            if c == contrast
        ]
        for st in _pack_readout_type_names(session, target)
    }


def moving_bar_window_t_rel(t0, t_on: int, win: int):
    """Window index into post-``t_on`` trace columns (numpy).

    Shared by ``FiveCol_MedSim_Pytorch._window_time_traces`` and
    ``plot.moving_bar._windows_by_batch``: slot ``k`` uses trace column
    ``t0 - t_on + k``; slots with ``t_rel < 0`` are pre-stimulus (masked zero).
    """
    t0 = np.asarray(t0, dtype=np.int64)
    win_ix = np.arange(int(win), dtype=np.int64)
    t_rel = t0[..., None] - int(t_on) + win_ix
    return t_rel, t_rel < 0


def moving_bar_window_t_rel_torch(t0, t_on: int, win: int, *, device=None):
    """Torch counterpart of :func:`moving_bar_window_t_rel`."""
    if not torch.is_tensor(t0):
        t0 = torch.as_tensor(t0, dtype=torch.long, device=device)
    else:
        t0 = t0.to(dtype=torch.long)
    if device is None:
        device = t0.device
    win_ix = torch.arange(int(win), dtype=torch.long, device=device)
    t_rel = t0[..., None] - int(t_on) + win_ix
    return t_rel, t_rel < 0


def resolve_i_baseline(value: Optional[float] = None) -> float:
    """Canonical photoreceptor baseline current (pA)."""
    return float(I_BASELINE if value is None else value)


@dataclass
class MovingBarT0Grids:
    t0_bn: np.ndarray
    before_steps: Dict[str, int]
    after_steps: Dict[str, int]


def column_first_stim_steps(
    column_current: np.ndarray,
    batch_idx: int,
    col_idxs: Sequence[int],
    i_baseline: float,
) -> List[int]:
    return [
        column_first_stim_step(column_current[batch_idx, :, col_idx], i_baseline=i_baseline)
        for col_idx in col_idxs
    ]


def moving_bar_spec_horizon(t_first_stis: Sequence[int], maxtime: int) -> Tuple[int, int, int]:
    """Return ``(fb, before_steps, after_steps)`` for one spec over all columns."""
    fb = int(min(t_first_stis))
    before = fb
    after = int(maxtime) - int(max(t_first_stis))
    return fb, before, after


def moving_bar_network_t0_bn(C, filt_cols: Sequence[StiColumn], n_batch: int, t0_map: dict) -> np.ndarray:
    """Expand per-column ``t0`` values to a full unit grid ``(B, N_units)``."""
    u_np = np.asarray(C.u, dtype=np.int64)
    v_np = np.asarray(C.v, dtype=np.int64)
    t0_bn = np.full((n_batch, C.n_units), -1, dtype=np.int64)
    for bi in range(n_batch):
        for c in filt_cols:
            t0 = t0_map.get((bi, int(c.u), int(c.v)))
            if t0 is None:
                continue
            on_col = (u_np == int(c.u)) & (v_np == int(c.v))
            t0_bn[bi, on_col] = t0
    return t0_bn


def build_moving_bar_t0_grids(
    column_current: np.ndarray,
    specs: Sequence[MovingBarSpec],
    maxtime: int,
    i_baseline: float,
    *,
    all_col_idxs: Sequence[int],
    filt_col_idxs: Sequence[int],
    network_C=None,
    filt_network_cols: Optional[Sequence[StiColumn]] = None,
    borst_filt_col_ids: Optional[Sequence[int]] = None,
) -> MovingBarT0Grids:
    """Plot/training-aligned ``t0`` grid and per-spec full horizons."""
    before_steps: Dict[str, int] = {}
    after_steps: Dict[str, int] = {}
    n_batch = len(specs)
    i_baseline = resolve_i_baseline(i_baseline)

    if network_C is not None:
        if filt_network_cols is None:
            raise ValueError("filt_network_cols required for network t0 grids")
        t0_map: dict = {}
        for bi, spec in enumerate(specs):
            t_first_all = column_first_stim_steps(
                column_current, bi, all_col_idxs, i_baseline,
            )
            fb, before, after = moving_bar_spec_horizon(t_first_all, maxtime)
            before_steps[spec.name] = before
            after_steps[spec.name] = after
            t_first_filt = column_first_stim_steps(
                column_current, bi, filt_col_idxs, i_baseline,
            )
            for c, tc in zip(filt_network_cols, t_first_filt):
                t0_map[(bi, int(c.u), int(c.v))] = tc - fb
        t0_bn = moving_bar_network_t0_bn(network_C, filt_network_cols, n_batch, t0_map)
    else:
        if borst_filt_col_ids is None:
            raise ValueError("borst_filt_col_ids required for Borst t0 grids")
        t0_bn = np.full((n_batch, ml.n_state_units()), -1, dtype=np.int64)
        for bi, spec in enumerate(specs):
            t_first_all = column_first_stim_steps(
                column_current, bi, all_col_idxs, i_baseline,
            )
            fb, before, after = moving_bar_spec_horizon(t_first_all, maxtime)
            before_steps[spec.name] = before
            after_steps[spec.name] = after
            t_first_filt = column_first_stim_steps(
                column_current, bi, borst_filt_col_ids, i_baseline,
            )
            for col_id, tc in zip(borst_filt_col_ids, t_first_filt):
                t0_bn[bi, ml.column_slice(col_id)] = tc - fb

    return MovingBarT0Grids(t0_bn=t0_bn, before_steps=before_steps, after_steps=after_steps)


def borst_sti_columns_scatter() -> List[StiColumn]:
    """Borst sti columns with photoreceptor unit indices for scatter."""
    out: List[StiColumn] = []
    for col in borst_sti_columns():
        pr = np.asarray(ml.photoreceptor_slice(col.col), dtype=np.int64)
        out.append(StiColumn(
            u=col.u,
            v=col.v,
            x=col.x,
            y=col.y,
            x_deg=col.x_deg,
            y_deg=col.y_deg,
            hex_xy=col.hex_xy,
            unit_idx=pr,
        ))
    return out


def _column_unit_map(columns: Sequence[StiColumn]) -> Tuple[np.ndarray, np.ndarray]:
    """Flat (col_idx, unit_idx) pairs for scattering column current onto units."""
    col_idx: List[int] = []
    unit_idx: List[int] = []
    for j, col in enumerate(columns):
        for u in np.asarray(col.unit_idx).ravel():
            col_idx.append(j)
            unit_idx.append(int(u))
    return (
        np.asarray(col_idx, dtype=np.int64),
        np.asarray(unit_idx, dtype=np.int64),
    )


def scatter_column_current(
    column_current: np.ndarray,
    columns: Sequence[StiColumn],
    n_units: int,
) -> np.ndarray:
    """Broadcast column current ``(T, n_cols)`` to unit current ``(T, n_units)``."""
    t_steps = column_current.shape[0]
    out = np.zeros((t_steps, n_units), dtype=np.float64)
    col_idx, unit_idx = _column_unit_map(columns)
    if len(col_idx):
        out[:, unit_idx] = column_current[:, col_idx]
    return out


def scatter_column_current_batched(
    column_current: np.ndarray,
    columns: Sequence[StiColumn],
    n_units: int,
) -> np.ndarray:
    """Broadcast ``(B, T, n_cols)`` column current to ``(B, T, n_units)``."""
    n_batch, t_steps, _ = column_current.shape
    out = np.zeros((n_batch, t_steps, n_units), dtype=np.float64)
    col_idx, unit_idx = _column_unit_map(columns)
    if len(col_idx):
        out[:, :, unit_idx] = column_current[:, :, col_idx]
    return out


def _column_uv(columns: Sequence[StiColumn]) -> List[Tuple[int, int]]:
    return [(c.u, c.v) for c in columns]


def _spec_contrast_set(specs: Sequence[MovingBarSpec]) -> frozenset:
    return frozenset(s.contrast for s in specs)


def _moving_bar_cache_key(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    column_uv: Sequence[Tuple[int, int]],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
    i_baseline: float,
    bar_extent: int,
    multi_bar: bool = True,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> str:
    stat = network_json.stat()
    payload = {
        "network": str(network_json.resolve()),
        "network_mtime_ns": stat.st_mtime_ns,
        "network_size": stat.st_size,
        "column_uv": list(column_uv),
        "bar_extent": int(bar_extent),
        "multi_bar": bool(multi_bar),
        "specs": [
            {
                "direction": s.direction,
                "contrast": s.contrast,
                "width_deg": s.width_deg,
                "speed_deg_s": s.speed_deg_s,
            }
            for s in specs
        ],
        "maxtime": maxtime,
        "t_on": t_on,
        "deltat_ms": deltat_ms,
        "i_baseline": i_baseline,
    }
    if i_bright_bar is not None:
        payload["i_bright_bar"] = i_bright_bar
    if i_dark_bar is not None:
        payload["i_dark_bar"] = i_dark_bar
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _moving_bar_cache_path(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    column_uv: Sequence[Tuple[int, int]],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
    i_baseline: float,
    bar_extent: int,
    multi_bar: bool = True,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> Path:
    key = _moving_bar_cache_key(
        network_json, specs, column_uv, maxtime, t_on, deltat_ms,
        i_baseline, bar_extent, multi_bar, i_bright_bar, i_dark_bar,
    )
    return moving_bar_cache_dir(network_json) / f"{key}.npz"


def _load_moving_bar_column_cache(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["column_current"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Ignoring corrupt moving-bar cache %s: %s", path, exc)
        return None


def _save_moving_bar_column_cache(path: Path, column_current: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, column_current=column_current)
    logger.info("Cached moving-bar column current to %s", path)


def build_moving_bar_signals(
    C,
    specs: Optional[Sequence[MovingBarSpec]] = None,
    maxtime: Optional[int] = None,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    bar_extent: int = DEFAULT_BAR_EXTENT,
    multi_bar: bool = True,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
) -> MovingBarStimulus:
    """Build batched photoreceptor current for moving-bar stimuli (network connectome).

    Returns ``signal`` with shape ``(B, T, N_units)`` where ``B = len(specs)``
    (16 by default).

    - ``multi_bar=True``: each batch row superposes simultaneous lane bars tiled with
      ``bar_extent`` spacing; bars are absent (baseline current) before lane entry
      and after lane exit.
    - ``multi_bar=False``: whole-field single-bar geometry over the full connectome
      sti field (no lane tiling/clipping).
    """
    device = device or C.device
    bar_extent = int(bar_extent)
    multi_bar = bool(multi_bar)
    specs = list(specs if specs is not None else gruntman_moving_bar_specs())
    contrasts = _spec_contrast_set(specs)
    i_bright = None
    i_dark = None
    if "bright" in contrasts:
        i_bright = I_BRIGHT if i_bright_bar is None else float(i_bright_bar)
    if "dark" in contrasts:
        i_dark = I_DARK if i_dark_bar is None else float(i_dark_bar)
    sti_cols = sti_columns(C)
    field_deg = field_bounds(sti_cols)
    if maxtime is None:
        maxtime = moving_bar_maxtime(
            specs, field_deg, bar_extent,
            multi_bar=multi_bar,
            t_on=t_on, deltat_ms=deltat_ms,
        )
    n_batch = len(specs)
    n_units = C.n_units
    sweep_end = moving_bar_sweep_end_step(
        specs, field_deg, bar_extent,
        multi_bar=multi_bar,
        t_on=t_on, deltat_ms=deltat_ms,
    )
    sweep_steps = sweep_end - t_on
    tail_steps = maxtime - sweep_end

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(C, "source_json", None)
    column_uv = _column_uv(sti_cols)
    if source_json is not None:
        cache_path = _moving_bar_cache_path(
            source_json, specs, column_uv, maxtime, t_on, deltat_ms,
            i_baseline, bar_extent, multi_bar, i_bright, i_dark,
        )

    col_curr: Optional[np.ndarray] = None
    if cache_path is not None and use_cache and not refresh_cache:
        col_curr = _load_moving_bar_column_cache(cache_path)
        if col_curr is not None:
            logger.info("Loaded moving-bar column current from cache %s", cache_path)

    if col_curr is None:
        col_curr = build_batched_column_current(
            sti_cols, specs, maxtime=maxtime, bar_extent=bar_extent,
            multi_bar=multi_bar,
            t_on=t_on, deltat_ms=deltat_ms,
            i_baseline=i_baseline,
            i_bright_bar=i_bright,
            i_dark_bar=i_dark,
        )
        if cache_path is not None and use_cache:
            _save_moving_bar_column_cache(cache_path, col_curr)

    signal_np = scatter_column_current_batched(col_curr, sti_cols, n_units)

    info = {
        "n_batch": n_batch,
        "n_sti_columns": len(sti_cols),
        "bar_extent": bar_extent,
        "multi_bar": multi_bar,
        "field_deg": field_deg,
        "maxtime": maxtime,
        "t_on": t_on,
        "sweep_end": sweep_end,
        "sweep_steps": sweep_steps,
        "sweep_time_s": sweep_steps * deltat_ms / 1000.0,
        "tail_steps": tail_steps,
        "tail_time_s": tail_steps * deltat_ms / 1000.0,
        "i_baseline": i_baseline,
        "speed_deg_s": specs[0].speed_deg_s if specs else GRUNTMAN_SPEED_DEG_S,
        "spec_names": [s.name for s in specs],
    }
    if i_bright is not None:
        info["i_bright_bar"] = i_bright
    if i_dark is not None:
        info["i_dark_bar"] = i_dark
    return MovingBarStimulus(
        signal=torch.as_tensor(signal_np, dtype=sim_dtype, device=device),
        column_current=col_curr,
        specs=specs,
        info=info,
    )


def _pd_nd_index(pd_nd: str) -> int:
    return PD_IDX if pd_nd == "PD" else ND_IDX


@dataclass
class MovingBarTarget:
    signal: torch.Tensor          # (B, T, N)
    data: torch.Tensor            # (n_cost, COST_WINDOW)
    power: torch.Tensor           # scalar
    cost_weight: torch.Tensor     # (n_cost,)
    cost_t0: torch.Tensor         # (n_cost,) absolute simulation step
    readout_batch: torch.Tensor   # (n_cost,) long
    readout_unit: torch.Tensor    # (n_cost,) long
    cost_pd_nd: torch.Tensor      # (n_cost,) long; 0=PD, 1=ND
    n_batch: int
    maxtime: int
    info: dict
    dsi_pos_rows: Optional[torch.Tensor] = None  # flat cost-row idx (right|up)
    dsi_neg_rows: Optional[torch.Tensor] = None  # flat cost-row idx (left|down)
    dsi_pos_ptr: Optional[torch.Tensor] = None   # (n_dsi+1,) CSR
    dsi_neg_ptr: Optional[torch.Tensor] = None   # (n_dsi+1,) CSR
    dsi_target: Optional[torch.Tensor] = None    # (n_dsi,)
    dsi_weight: Optional[torch.Tensor] = None    # (n_dsi,)
    dsi_power: Optional[torch.Tensor] = None     # scalar


def _fig1_trace_ids(npz_path: Path) -> List[str]:
    with np.load(npz_path) as d:
        return sorted({k.replace("__time_ms", "") for k in d.files if k.endswith("__time_ms")})


def load_fig1_trace(
    trace_id: str,
    npz_path: Path = FIG1_CI_NPZ,
    deltat_ms: float = DELTAT_MS,
) -> np.ndarray:
    """Resample one fig1 trace onto the moving-bar cost window."""
    n_steps = ms_to_steps(COST_WINDOW_MS, deltat_ms=deltat_ms) + 1
    before_steps = ms_to_steps(COST_ALIGNED_FIRST_STI_MS, deltat_ms=deltat_ms)
    key = (
        f"{trace_id}|{n_steps}|{before_steps}|{deltat_ms}"
        f"|{COST_WINDOW_MS}|{COST_ALIGNED_FIRST_STI_MS}"
    )
    if key in _TRACE_CACHE:
        return _TRACE_CACHE[key]

    with np.load(npz_path) as d:
        t_key, v_key = f"{trace_id}__time_ms", f"{trace_id}__vm_mv"
        if t_key not in d.files:
            raise KeyError(f"missing trace {trace_id!r} in {npz_path}")
        time_ms = np.asarray(d[t_key], dtype=np.float64)
        vm_mv = np.asarray(d[v_key], dtype=np.float64)

    query_ms = np.arange(n_steps, dtype=np.float64) * deltat_ms
    trace = np.interp(query_ms, time_ms, vm_mv, left=vm_mv[0], right=vm_mv[-1])
    _TRACE_CACHE[key] = trace
    return trace


def load_fig1_traces(
    npz_path: Path = FIG1_CI_NPZ,
    deltat_ms: float = DELTAT_MS,
) -> Dict[str, np.ndarray]:
    """All fig1 traces resampled to the per-column training window."""
    return {
        tid: load_fig1_trace(tid, npz_path, deltat_ms)
        for tid in _fig1_trace_ids(npz_path)
    }


def col2subtype(C, u: int, v: int, subtype: str, names: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit indices of ``subtype`` (e.g. ``T4a``) on column ``(u, v)``."""
    if names is None:
        names = unit_type_names(C)
    return np.where(
        (C.u == int(u)) & (C.v == int(v)) & (names == subtype),
    )[0]


def moving_bar_units_on_columns(C, cell: str, cols: Sequence) -> np.ndarray:
    """Unit indices of ``cell`` on any of ``cols`` (vectorized axial uv pack)."""
    if not cols:
        return np.zeros(0, dtype=np.int64)
    if cell not in C.type_names:
        raise ValueError(f"unknown cell type {cell!r}; known: {list(C.type_names)}")
    ti = int(C.type_names.index(cell))
    u_np, v_np = network_uv_np(C)
    type_ids = _as_int64_np(C.node_type)
    uv_span = int(max(np.max(np.abs(u_np)), np.max(np.abs(v_np)), 1)) + 1
    pack = (u_np + uv_span) * (2 * uv_span + 1) + (v_np + uv_span)
    col_pack = np.array(
        [
            (int(c.u) + uv_span) * (2 * uv_span + 1) + (int(c.v) + uv_span)
            for c in cols
        ],
        dtype=np.int64,
    )
    return np.where((type_ids == ti) & np.isin(pack, col_pack))[0].astype(np.int64)


def filter_requested_specs(
    available: Sequence[str],
    requested: Optional[Sequence[str]],
) -> List[str]:
    """Keep ``requested`` specs that exist in ``available``; omit ``requested`` → all."""
    avail = list(available)
    if requested is None:
        return avail
    missing = [s for s in requested if s not in avail]
    if missing:
        raise ValueError(f"spec(s) {missing} not in {avail}")
    return list(requested)


def _assemble_moving_bar_readouts(
    *,
    specs: Sequence[MovingBarSpec],
    column_current: np.ndarray,
    cost_col_idxs: Sequence[int],
    i_baseline: float,
    before_steps: int,
    after_steps: int,
    maxtime: int,
    side: str,
    fig1: Dict[str, np.ndarray],
    present: Sequence[str],
    units_for_col_subtype: Callable[[int, int, str], Sequence[int]],
) -> Tuple[
    List[int], List[int], List[str], List[np.ndarray], List[float], List[int], List[int], int,
]:
    r_batch, r_unit, r_subtype, r_target, r_weight, r_t0, r_pd_nd = (
        [], [], [], [], [], [], [],
    )
    skipped_orthogonal = 0
    i_baseline = resolve_i_baseline(i_baseline)
    for b, spec in enumerate(specs):
        for col_idx in cost_col_idxs:
            t_first_sti = column_first_stim_step(
                column_current[b, :, col_idx], i_baseline=i_baseline,
            )
            t0 = t_first_sti - before_steps
            if t0 < 0 or t_first_sti + after_steps > maxtime:
                raise ValueError(
                    f"cost window out of range for column index {col_idx} "
                    f"spec={spec.name}: t_first_sti={t_first_sti}, maxtime={maxtime}"
                )
            for subtype in present:
                pref = motion_preference(side, subtype, spec.direction, spec.contrast)
                if pref is None:
                    skipped_orthogonal += 1
                    continue
                trace_id = fig1_key_for_stimulus(side, subtype, spec)
                if trace_id not in fig1:
                    raise KeyError(f"fig1 trace missing: {trace_id}")
                units = units_for_col_subtype(b, col_idx, subtype)
                if len(units) == 0:
                    continue
                target = fig1[trace_id]
                pd_nd_idx = _pd_nd_index(pref.pd_nd)
                for uidx in units:
                    r_batch.append(b)
                    r_unit.append(int(uidx))
                    r_subtype.append(str(subtype))
                    r_target.append(target)
                    r_weight.append(1.0)
                    r_t0.append(t0)
                    r_pd_nd.append(pd_nd_idx)
    return (
        r_batch, r_unit, r_subtype, r_target, r_weight, r_t0, r_pd_nd,
        skipped_orthogonal,
    )


def _pack_moving_bar_readout_tensors(
    r_batch, r_unit, r_target, r_weight, r_t0, r_pd_nd, *, device, sim_dtype,
):
    data = torch.tensor(np.asarray(r_target), dtype=sim_dtype, device=device)
    cost_weight = torch.tensor(np.asarray(r_weight), dtype=sim_dtype, device=device)
    readout_batch = torch.tensor(np.asarray(r_batch), dtype=torch.long, device=device)
    readout_unit = torch.tensor(np.asarray(r_unit), dtype=torch.long, device=device)
    cost_t0 = torch.tensor(np.asarray(r_t0), dtype=torch.long, device=device)
    cost_pd_nd = torch.tensor(np.asarray(r_pd_nd), dtype=torch.long, device=device)
    power = torch.sum(cost_weight[:, None] * data ** 2)
    if float(power) == 0.0:
        power = torch.tensor(1.0, dtype=sim_dtype, device=device)
    return data, cost_weight, readout_batch, readout_unit, cost_t0, cost_pd_nd, power


def _moving_bar_peak_kwargs(
    contrasts: Sequence[str],
    *,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> dict:
    """Peak-current kwargs for specs that only include the relevant contrasts."""
    contrast_set = frozenset(contrasts)
    kw = {}
    if "bright" in contrast_set and i_bright_bar is not None:
        kw["i_bright_bar"] = float(i_bright_bar)
    if "dark" in contrast_set and i_dark_bar is not None:
        kw["i_dark_bar"] = float(i_dark_bar)
    return kw


def _present_readout_subtypes(
    readout_subtypes: Optional[Sequence[str]],
    default_pool: Sequence[str],
    available: Sequence[str],
    *,
    context: str,
) -> List[str]:
    pool = tuple(readout_subtypes) if readout_subtypes is not None else tuple(default_pool)
    present = [st for st in pool if st in set(available)]
    if not present:
        raise ValueError(
            f"{context} has no moving-bar readout subtypes "
            f"(requested {list(pool)!r})",
        )
    return present


def build_moving_bar_target(
    C,
    device: Optional[str] = None,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    fig1_path: Path = FIG1_CI_NPZ,
    use_cache: bool = True,
    bar_extent: int = DEFAULT_BAR_EXTENT,
    multi_bar: bool = True,
    cost_extent: Optional[int] = None,
    i_baseline: Optional[float] = None,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
    readout_subtypes: Optional[Sequence[str]] = None,
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
) -> MovingBarTarget:
    """Build multi-bar stimulus + fig1 targets for sti columns × T4/T5 subtypes.

    ``bar_extent`` sets per-lane spacing (hex-column units). ``cost_extent``
    restricts cost to columns inside the central hex disc (``None`` = all sti
    columns). Stimulus still drives all photoreceptors.
    """
    device = device or C.device
    side = normalize_side(C.meta.get("side", "right"))

    specs = gruntman_moving_bar_specs(contrasts=tuple(contrasts))
    contrast_set = frozenset(contrasts)
    i_baseline_val = resolve_i_baseline(i_baseline)
    stim = build_moving_bar_signals(
        C, specs=specs, t_on=t_on, deltat_ms=deltat_ms,
        bar_extent=bar_extent, multi_bar=bool(multi_bar),
        device=device, use_cache=use_cache,
        network_json=getattr(C, "source_json", None),
        i_baseline=i_baseline_val,
        sim_dtype=sim_dtype,
        **_moving_bar_peak_kwargs(
            contrast_set,
            i_bright_bar=i_bright_bar,
            i_dark_bar=i_dark_bar,
        ),
    )
    maxtime = int(stim.info["maxtime"])
    field_deg = stim.info["field_deg"]
    fig1 = load_fig1_traces(fig1_path, deltat_ms=deltat_ms)
    before_steps = ms_to_steps(COST_ALIGNED_FIRST_STI_MS, deltat_ms=deltat_ms)
    after_steps = ms_to_steps(COST_WINDOW_AFTER_MS, deltat_ms=deltat_ms)
    win_steps = ms_to_steps(COST_WINDOW_MS, deltat_ms=deltat_ms) + 1

    present = _present_readout_subtypes(
        readout_subtypes, READOUT_SUBTYPES, C.type_names,
        context="network",
    )

    type_names = unit_type_names(C)
    sti_cols = sti_columns(C)
    uv_to_col_idx = {(int(c.u), int(c.v)): j for j, c in enumerate(sti_cols)}
    cols = moving_bar_cost_columns(C, cost_extent=cost_extent)
    center_col = cols[0] if cost_extent == 0 and len(cols) == 1 else None
    cost_col_idxs = [uv_to_col_idx[(int(c.u), int(c.v))] for c in cols]
    col_by_idx = {idx: c for c, idx in zip(cols, cost_col_idxs)}

    def _units_for_col_subtype(_b, col_idx, subtype):
        col = col_by_idx[col_idx]
        return col2subtype(C, col.u, col.v, subtype, type_names)

    rows = _assemble_moving_bar_readouts(
        specs=stim.specs,
        column_current=stim.column_current,
        cost_col_idxs=cost_col_idxs,
        i_baseline=i_baseline_val,
        before_steps=before_steps,
        after_steps=after_steps,
        maxtime=maxtime,
        side=side,
        fig1=fig1,
        present=present,
        units_for_col_subtype=_units_for_col_subtype,
    )
    (
        r_batch, r_unit, r_subtype, r_target, r_weight, r_t0, r_pd_nd,
        skipped_orthogonal,
    ) = rows

    if not r_batch:
        raise ValueError("no moving-bar cost cells (check subtypes and sti columns)")

    data, cost_weight, readout_batch, readout_unit, cost_t0, cost_pd_nd, power = (
        _pack_moving_bar_readout_tensors(
            r_batch, r_unit, r_target, r_weight, r_t0, r_pd_nd,
            device=device, sim_dtype=sim_dtype,
        )
    )
    from analysis.DSI import build_dsi_pack_fields

    (
        dsi_pos_rows, dsi_neg_rows, dsi_pos_ptr, dsi_neg_ptr,
        dsi_tgt, dsi_w, dsi_pow,
    ) = build_dsi_pack_fields(
        stim.specs, r_batch, r_subtype, r_target, r_weight,
        device=device, sim_dtype=sim_dtype,
    )

    info = {
        **stim.info,
        "n_cost": int(data.shape[0]),
        "n_cost_pd": int((cost_pd_nd == PD_IDX).sum().item()),
        "n_cost_nd": int((cost_pd_nd == ND_IDX).sum().item()),
        "n_cost_dsi": int(dsi_tgt.shape[0]),
        "n_batch": stim.info["n_batch"],
        "n_cost_columns": len(cols),
        "cost_extent": cost_extent,
        "cost_column_uv": (int(center_col.u), int(center_col.v)) if center_col else None,
        "side": side,
        "present_subtypes": present,
        "skipped_orthogonal": skipped_orthogonal,
        "fig1_path": str(fig1_path),
        "cost_window_before_ms": COST_WINDOW_BEFORE_MS,
        "cost_window_after_ms": COST_WINDOW_AFTER_MS,
        "cost_window_ms": COST_WINDOW_MS,
        "cost_aligned_first_sti_ms": COST_ALIGNED_FIRST_STI_MS,
        "cost_window_steps": win_steps,
        "deltat_ms": float(deltat_ms),
    }
    return MovingBarTarget(
        signal=stim.signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        cost_pd_nd=cost_pd_nd,
        n_batch=stim.info["n_batch"],
        maxtime=maxtime,
        info=info,
        dsi_pos_rows=dsi_pos_rows,
        dsi_neg_rows=dsi_neg_rows,
        dsi_pos_ptr=dsi_pos_ptr,
        dsi_neg_ptr=dsi_neg_ptr,
        dsi_target=dsi_tgt,
        dsi_weight=dsi_w,
        dsi_power=dsi_pow,
    )


def build_borst_moving_bar_target(
    device: Optional[str] = None,
    t_on: int = T_ON,
    deltat_ms: float = DELTAT_MS,
    fig1_path: Path = FIG1_CI_NPZ,
    readout_subtypes: Optional[Sequence[str]] = None,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    contrasts: Sequence[str] = ("bright", "dark"),
    sim_dtype: torch.dtype = SIM_DTYPE_DEFAULT,
) -> MovingBarTarget:
    """Build Borst 5-column horizontal moving-bar target for T4/T5 subtypes."""
    device = device or "cpu"
    i_baseline = resolve_i_baseline(i_baseline)
    specs = gruntman_moving_bar_specs(directions=("right", "left"), contrasts=tuple(contrasts))
    cols = list(borst_sti_columns())
    scatter_cols = borst_sti_columns_scatter()
    field_deg = field_bounds(cols)
    maxtime = whole_field_moving_bar_maxtime(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    sweep_end = whole_field_moving_bar_sweep_end_step(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    fig1 = load_fig1_traces(fig1_path, deltat_ms=deltat_ms)
    before_steps = ms_to_steps(COST_ALIGNED_FIRST_STI_MS, deltat_ms=deltat_ms)
    after_steps = ms_to_steps(COST_WINDOW_AFTER_MS, deltat_ms=deltat_ms)
    win_steps = ms_to_steps(COST_WINDOW_MS, deltat_ms=deltat_ms) + 1
    present = _present_readout_subtypes(
        readout_subtypes, BORST_READOUT_SUBTYPES, ml.ctype.tolist(),
        context="Borst",
    )

    contrast_set = frozenset(contrasts)
    col_kw = dict(i_baseline=i_baseline)
    if "bright" in contrast_set:
        col_kw["i_bright_bar"] = I_BRIGHT if i_bright_bar is None else float(i_bright_bar)
    if "dark" in contrast_set:
        col_kw["i_dark_bar"] = I_DARK if i_dark_bar is None else float(i_dark_bar)
    column_current = borst_whole_field_batched_column_current(
        cols, specs, maxtime, t_on=t_on, deltat_ms=deltat_ms,
        **col_kw,
    )
    n_units = ml.n_state_units()
    signal_np = scatter_column_current_batched(column_current, scatter_cols, n_units)
    signal = torch.as_tensor(signal_np, dtype=sim_dtype, device=device)

    cost_col_ids = list(range(ml.nofcols))

    def _units_for_col_subtype(_b, col_id, subtype):
        return [int(ml.unit_index(col_id, ml.type_index(subtype)))]

    rows = _assemble_moving_bar_readouts(
        specs=specs,
        column_current=column_current,
        cost_col_idxs=cost_col_ids,
        i_baseline=i_baseline,
        before_steps=before_steps,
        after_steps=after_steps,
        maxtime=maxtime,
        side="right",
        fig1=fig1,
        present=present,
        units_for_col_subtype=_units_for_col_subtype,
    )
    (
        r_batch, r_unit, r_subtype, r_target, r_weight, r_t0, r_pd_nd,
        skipped_orthogonal,
    ) = rows

    if not r_batch:
        raise ValueError("no Borst moving-bar cost cells")

    data, cost_weight, readout_batch, readout_unit, cost_t0, cost_pd_nd, power = (
        _pack_moving_bar_readout_tensors(
            r_batch, r_unit, r_target, r_weight, r_t0, r_pd_nd,
            device=device, sim_dtype=sim_dtype,
        )
    )
    from analysis.DSI import build_dsi_pack_fields

    (
        dsi_pos_rows, dsi_neg_rows, dsi_pos_ptr, dsi_neg_ptr,
        dsi_tgt, dsi_w, dsi_pow,
    ) = build_dsi_pack_fields(
        specs, r_batch, r_subtype, r_target, r_weight,
        device=device, sim_dtype=sim_dtype,
    )

    info = {
        "n_cost": int(data.shape[0]),
        "n_cost_pd": int((cost_pd_nd == PD_IDX).sum().item()),
        "n_cost_nd": int((cost_pd_nd == ND_IDX).sum().item()),
        "n_cost_dsi": int(dsi_tgt.shape[0]),
        "n_batch": len(specs),
        "n_cost_columns": len(cols),
        "cost_extent": None,
        "cost_column_uv": None,
        "side": "right",
        "present_subtypes": present,
        "skipped_orthogonal": skipped_orthogonal,
        "fig1_path": str(fig1_path),
        "cost_window_before_ms": COST_WINDOW_BEFORE_MS,
        "cost_window_after_ms": COST_WINDOW_AFTER_MS,
        "cost_window_ms": COST_WINDOW_MS,
        "cost_aligned_first_sti_ms": COST_ALIGNED_FIRST_STI_MS,
        "cost_window_steps": win_steps,
        "deltat_ms": float(deltat_ms),
        "maxtime": maxtime,
        "t_on": t_on,
        "field_deg": field_deg,
        "sweep_steps": sweep_end - t_on,
        "sweep_time_s": (sweep_end - t_on) * (deltat_ms / 1000.0),
        "spec_names": [s.name for s in specs],
        "n_sti_columns": ml.nofcols,
        "mode": "borst",
        "i_baseline": i_baseline,
    }
    if "bright" in contrast_set:
        info["i_bright_bar"] = col_kw["i_bright_bar"]
    if "dark" in contrast_set:
        info["i_dark_bar"] = col_kw["i_dark_bar"]
    return MovingBarTarget(
        signal=signal,
        data=data,
        power=power,
        cost_weight=cost_weight,
        cost_t0=cost_t0,
        readout_batch=readout_batch,
        readout_unit=readout_unit,
        cost_pd_nd=cost_pd_nd,
        n_batch=len(specs),
        maxtime=maxtime,
        info=info,
        dsi_pos_rows=dsi_pos_rows,
        dsi_neg_rows=dsi_neg_rows,
        dsi_pos_ptr=dsi_pos_ptr,
        dsi_neg_ptr=dsi_neg_ptr,
        dsi_target=dsi_tgt,
        dsi_weight=dsi_w,
        dsi_power=dsi_pow,
    )
