# -*- coding: utf-8 -*-
"""Load visual stimuli onto a connectome as per-unit input current.

Column geometry and stimulus physics live in :mod:`visual_stimulus` (e.g.
``moving_bar_stimulus``). This module maps column-level currents to photoreceptor
units on a loaded :class:`network.construction.Network`.

``build_moving_bar_signals`` returns ``signal`` with shape ``(B, T, N_units)``,
ready to assign to ``FiveCol_MedSim_Pytorch.signal``. Default ``T`` is
``t_on`` (0.5 s baseline) plus the steps needed for the bar to sweep the
photo field, not the global Borst ``IMPULSE_MAXTIME``.
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

import connectome_path  # noqa: F401

from column_mapper import DEFAULT_KERNEL_SIZE, hex_to_pixel
from connectome_io import moving_bar_cache_dir
from Medulla_Library import SIGNAL_BASELINE, SIGNAL_BRIGHT, SIGNAL_DARK, T_ON
from visual_stimulus.moving_bar_stimulus import (
    GRUNTMAN_SPEED_DEG_S,
    HexColumn,
    MovingBarSpec,
    build_batched_column_current,
    field_bounds,
    gruntman_moving_bar_specs,
    hex_vertices,
    moving_bar_maxtime,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MovingBarSpec",
    "MovingBarStimulus",
    "PhotoColumn",
    "build_moving_bar_signals",
    "gruntman_moving_bar_specs",
]


@dataclass
class PhotoColumn:
    """One photo column on a connectome, with unit indices for scattering."""

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


def _photo_columns(C) -> List[PhotoColumn]:
    """One entry per axial (u, v) that has photoreceptor input units."""
    cols: Dict[Tuple[int, int], PhotoColumn] = {}
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
        cols[key] = PhotoColumn(
            u=key[0],
            v=key[1],
            x=x,
            y=y,
            hex_xy=hex_vertices(x, y),
            unit_idx=np.asarray(units, dtype=np.int64),
        )
    return [cols[k] for k in sorted(cols)]


def _as_hex_columns(columns: Sequence[PhotoColumn]) -> List[HexColumn]:
    return [
        HexColumn(u=c.u, v=c.v, x=c.x, y=c.y, hex_xy=c.hex_xy)
        for c in columns
    ]


def _column_unit_map(columns: Sequence[PhotoColumn]) -> Tuple[np.ndarray, np.ndarray]:
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
    columns: Sequence[PhotoColumn],
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
    columns: Sequence[PhotoColumn],
    n_units: int,
) -> np.ndarray:
    """Broadcast ``(B, T, n_cols)`` column current to ``(B, T, n_units)``."""
    n_batch, t_steps, _ = column_current.shape
    out = np.zeros((n_batch, t_steps, n_units), dtype=np.float64)
    col_idx, unit_idx = _column_unit_map(columns)
    if len(col_idx):
        out[:, :, unit_idx] = column_current[:, :, col_idx]
    return out


def _column_uv(columns: Sequence[PhotoColumn]) -> List[Tuple[int, int]]:
    return [(c.u, c.v) for c in columns]


def _moving_bar_cache_key(
    network_json: Path,
    specs: Sequence[MovingBarSpec],
    column_uv: Sequence[Tuple[int, int]],
    maxtime: int,
    t_on: int,
    deltat_ms: float,
    i_baseline: float,
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
        "signal_baseline": i_baseline,
        "signal_bright": SIGNAL_BRIGHT,
        "signal_dark": SIGNAL_DARK,
    }
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
) -> Path:
    key = _moving_bar_cache_key(
        network_json, specs, column_uv, maxtime, t_on, deltat_ms, i_baseline,
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
    i_baseline: float = SIGNAL_BASELINE,
    device: Optional[str] = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
    network_json: Optional[Path] = None,
) -> MovingBarStimulus:
    """Build batched photoreceptor current for moving-bar stimuli.

    Returns ``signal`` with shape ``(B, T, N_units)`` where ``B = len(specs)``
    (16 by default). Before ``t_on`` all currents are ``i_baseline``.

    Default ``maxtime`` is ``t_on`` plus the steps needed for the bar to sweep
    the photo field (not the global ``IMPULSE_MAXTIME`` used by Borst training).
    Pass an explicit ``maxtime`` to override.

    Column currents ``(B, T, n_cols)`` are cached under
    ``<network_run>/moving_bar_cache/`` (see ``connectome_io.moving_bar_cache_dir``).
    Unit scattering is cheap and always runs from cache or a fresh build.
    """
    device = device or C.device
    specs = list(specs if specs is not None else gruntman_moving_bar_specs())
    photo_cols = _photo_columns(C)
    hex_cols = _as_hex_columns(photo_cols)
    field_deg = field_bounds(hex_cols)
    if maxtime is None:
        maxtime = moving_bar_maxtime(specs, field_deg, t_on=t_on, deltat_ms=deltat_ms)
    n_batch = len(specs)
    n_units = C.n_units
    sweep_steps = maxtime - t_on

    cache_path: Optional[Path] = None
    source_json = Path(network_json) if network_json is not None else getattr(C, "source_json", None)
    column_uv = _column_uv(photo_cols)
    if source_json is not None:
        cache_path = _moving_bar_cache_path(
            source_json, specs, column_uv, maxtime, t_on, deltat_ms, i_baseline,
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
        )
        if cache_path is not None and use_cache:
            _save_moving_bar_column_cache(cache_path, col_curr)

    signal_np = scatter_column_current_batched(col_curr, photo_cols, n_units)
    signal_np[:, :t_on, :] = i_baseline

    info = {
        "n_batch": n_batch,
        "n_photo_columns": len(photo_cols),
        "field_deg": field_deg,
        "maxtime": maxtime,
        "t_on": t_on,
        "sweep_steps": sweep_steps,
        "sweep_time_s": sweep_steps * deltat_ms / 1000.0,
        "i_bright": SIGNAL_BRIGHT,
        "i_dark": SIGNAL_DARK,
        "i_baseline": i_baseline,
        "speed_deg_s": specs[0].speed_deg_s if specs else GRUNTMAN_SPEED_DEG_S,
        "spec_names": [s.name for s in specs],
    }
    return MovingBarStimulus(
        signal=torch.as_tensor(signal_np, dtype=torch.float64, device=device),
        column_current=col_curr,
        specs=specs,
        info=info,
    )
