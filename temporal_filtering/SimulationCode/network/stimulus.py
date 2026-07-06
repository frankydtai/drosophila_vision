# -*- coding: utf-8 -*-
"""Load visual stimuli onto a connectome as per-unit input current.

Column geometry and stimulus physics live in :mod:`visual_stimulus` (e.g.
``moving_bar_stimulus``). This module maps column-level currents to photoreceptor
units on a loaded :class:`network.construction.Network`.

``build_moving_bar_signals`` returns ``signal`` with shape ``(B, T, N_units)``,
ready to assign to ``FiveCol_MedSim_Pytorch.signal``. Default ``T`` is
``t_on`` (0.5 s baseline) + sweep + moving-bar tail (``MOVING_BAR_TAIL_MS`` post-sweep baseline),
not the global Borst ``IMPULSE_MAXTIME``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import network_bootstrap  # noqa: F401

from column_mapper import DEFAULT_KERNEL_SIZE, hex_to_pixel, hex_vertices
from connectome_io import moving_bar_cache_dir
from Medulla_Library import I_BASELINE, I_BRIGHT, I_DARK, T_ON
from visual_stimulus.moving_bar_stimulus import (
    GRUNTMAN_SPEED_DEG_S,
    HexColumn,
    MovingBarSpec,
    build_batched_column_current,
    field_bounds,
    gruntman_moving_bar_specs,
    moving_bar_maxtime,
    moving_bar_sweep_end_step,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MovingBarSpec",
    "MovingBarStimulus",
    "StiColumn",
    "build_moving_bar_signals",
    "gruntman_moving_bar_specs",
]


@dataclass
class StiColumn:
    """One sti column on a connectome, with unit indices for scattering."""

    u: int
    v: int
    x: float
    y: float
    hex_xy: np.ndarray
    unit_idx: np.ndarray


@dataclass
class MovingBarStimulus:
    signal: torch.Tensor
    column_current: np.ndarray
    specs: List[MovingBarSpec]
    info: dict = field(default_factory=dict)


def _sti_columns(C) -> List[StiColumn]:
    """One entry per axial (u, v) that has photoreceptor input units."""
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
        x, y = hex_to_pixel(key[0], key[1], DEFAULT_KERNEL_SIZE)
        x = float(x)
        y = float(y)
        cols[key] = StiColumn(
            u=key[0],
            v=key[1],
            x=x,
            y=y,
            hex_xy=hex_vertices(x, y),
            unit_idx=np.asarray(units, dtype=np.int64),
        )
    return [cols[k] for k in sorted(cols)]


def sti_columns(C) -> List[StiColumn]:
    """Sti columns with photoreceptor units (one per axial ``(u, v)``)."""
    return _sti_columns(C)


def center_sti_column(C) -> StiColumn:
    """Sti column at hex origin ``(u, v) = (0, 0)``, or closest to field centre."""
    cols = sti_columns(C)
    if not cols:
        raise ValueError("network has no sti columns")
    for col in cols:
        if col.u == 0 and col.v == 0:
            return col
    return min(cols, key=lambda c: c.x * c.x + c.y * c.y)


def column_in_cost_extent(u, v, cost_extent=None) -> bool:
    """True when axial ``(u, v)`` lies in the cost hex disc (``None`` = all columns)."""
    if cost_extent is None:
        return True
    import column_mapper
    return bool(column_mapper.inside_mask(int(u), int(v), int(cost_extent)))


def cost_sti_columns(C, cost_extent=None) -> List[StiColumn]:
    """Sti columns used for moving-bar cost (optional central hex disc)."""
    cols = sti_columns(C)
    if cost_extent is None:
        return cols
    return [c for c in cols if column_in_cost_extent(c.u, c.v, cost_extent)]


def _as_hex_columns(columns: Sequence[StiColumn]) -> List[HexColumn]:
    return [
        HexColumn(u=c.u, v=c.v, x=c.x, y=c.y, hex_xy=c.hex_xy)
        for c in columns
    ]


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
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> str:
    stat = network_json.stat()
    payload = {
        "network": str(network_json.resolve()),
        "network_mtime_ns": stat.st_mtime_ns,
        "network_size": stat.st_size,
        "column_uv": list(column_uv),
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
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
) -> Path:
    key = _moving_bar_cache_key(
        network_json, specs, column_uv, maxtime, t_on, deltat_ms,
        i_baseline, i_bright_bar, i_dark_bar,
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
    deltat_ms: float = 10.0,
    i_baseline: float = I_BASELINE,
    i_bright_bar: Optional[float] = None,
    i_dark_bar: Optional[float] = None,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
) -> MovingBarStimulus:
    """Build batched photoreceptor current for moving-bar stimuli.

    Returns ``signal`` with shape ``(B, T, N_units)`` where ``B = len(specs)``
    (16 by default). Before ``t_on`` and after the sweep, all currents are
    ``i_baseline``; during the sweep they follow bar coverage (bright/dark).

    Default ``maxtime`` is ``t_on`` (0.5 s baseline) + sweep + tail
    (``MOVING_BAR_TAIL_MS`` via :func:`visual_stimulus.moving_bar_stimulus.moving_bar_maxtime`),
    not the global ``IMPULSE_MAXTIME`` used by Borst training. Pass an explicit
    ``maxtime`` to override.

    Column currents ``(B, T, n_cols)`` are cached under
    ``<network_run>/moving_bar_cache/`` (see ``connectome_io.moving_bar_cache_dir``).
    Unit scattering is cheap and always runs from cache or a fresh build.
    """
    device = device or C.device
    specs = list(specs if specs is not None else gruntman_moving_bar_specs())
    contrasts = _spec_contrast_set(specs)
    i_bright = None
    i_dark = None
    if "bright" in contrasts:
        i_bright = I_BRIGHT if i_bright_bar is None else float(i_bright_bar)
    if "dark" in contrasts:
        i_dark = I_DARK if i_dark_bar is None else float(i_dark_bar)
    sti_cols = _sti_columns(C)
    hex_cols = _as_hex_columns(sti_cols)
    field_deg = field_bounds(hex_cols)
    if maxtime is None:
        maxtime = moving_bar_maxtime(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    n_batch = len(specs)
    n_units = C.n_units
    sweep_end = moving_bar_sweep_end_step(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    sweep_steps = sweep_end - t_on
    tail_steps = maxtime - sweep_end

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(C, "source_json", None)
    column_uv = _column_uv(sti_cols)
    if source_json is not None:
        cache_path = _moving_bar_cache_path(
            source_json, specs, column_uv, maxtime, t_on, deltat_ms,
            i_baseline, i_bright, i_dark,
        )

    col_curr: Optional[np.ndarray] = None
    if cache_path is not None and use_cache and not refresh_cache:
        col_curr = _load_moving_bar_column_cache(cache_path)
        if col_curr is not None:
            logger.info("Loaded moving-bar column current from cache %s", cache_path)

    if col_curr is None:
        col_curr = build_batched_column_current(
            hex_cols, specs, maxtime=maxtime, t_on=t_on, deltat_ms=deltat_ms,
            i_baseline=i_baseline,
            i_bright_bar=i_bright,
            i_dark_bar=i_dark,
        )
        if cache_path is not None and use_cache:
            _save_moving_bar_column_cache(cache_path, col_curr)

    signal_np = scatter_column_current_batched(col_curr, sti_cols, n_units)
    signal_np[:, :t_on, :] = i_baseline

    info = {
        "n_batch": n_batch,
        "n_sti_columns": len(sti_cols),
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
        signal=torch.as_tensor(signal_np, dtype=torch.float64, device=device),
        column_current=col_curr,
        specs=specs,
        info=info,
    )
