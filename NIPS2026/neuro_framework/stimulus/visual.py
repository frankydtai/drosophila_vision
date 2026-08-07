"""
Visual Stimulus Library
=======================
Generates spatiotemporal visual stimuli for the optic lobe model.

All stimuli return tensors of shape  (batch, T, n_nodes)  (PyTorch)
or numpy arrays of the same shape.

Supported stimuli
-----------------
  FlashStimulus        – full-field luminance step
  MovingBarStimulus    – single bright bar sweeping across the hexagonal grid
  MovingEdgeStimulus   – luminance step edge moving across the grid
  SinusoidalGrating    – spatiotemporal sinusoidal grating
  NaturalSceneStimulus – placeholder for natural image sequences

Convention
----------
  * Pixel / node values are in [0, 1]  (relative luminance).
  * Time is in milliseconds; dt is the simulation timestep.
  * n_nodes matches the number of photoreceptor / input neurons.
    For a hexagonal grid of radius r, n_nodes = 3r(r-1)+1.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "FlashStimulus",
    "MovingBarStimulus",
    "MovingEdgeStimulus",
    "SinusoidalGrating",
    "hex_grid_coords",
    "build_stimulus_tensor",
]


# ---------------------------------------------------------------------------
# Hexagonal grid utilities
# ---------------------------------------------------------------------------

def hex_grid_coords(radius: int) -> np.ndarray:
    """
    Generate 2-D Cartesian coordinates for a hexagonal grid of given radius.

    Parameters
    ----------
    radius : int
        Number of rings (radius=1 gives 7 nodes, radius=5 gives 61, etc.).

    Returns
    -------
    coords : np.ndarray  shape (n_nodes, 2)
        (x, y) positions in normalised units where adjacent nodes are 1 apart.
    """
    coords = []
    for q in range(-radius, radius + 1):
        r_lo = max(-radius, -q - radius)
        r_hi = min(radius,  -q + radius)
        for r in range(r_lo, r_hi + 1):
            x = q + 0.5 * r
            y = r * (3 ** 0.5) / 2
            coords.append([x, y])
    return np.array(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseStimulus:
    """
    Abstract base class for visual stimuli.

    Subclasses implement ``_generate(t_array, coords)`` which returns
    a float32 numpy array of shape ``(T, n_nodes)``.
    """

    def __init__(
        self,
        coords: np.ndarray,        # (n_nodes, 2)
        t_start: float = 0.0,
        t_end:   float = 500.0,
        dt:      float = 1.0,
        background: float = 0.0,
    ):
        self.coords     = coords
        self.t_start    = t_start
        self.t_end      = t_end
        self.dt         = dt
        self.background = background
        self.t_array    = np.arange(t_start, t_end, dt, dtype=np.float32)
        self.T          = len(self.t_array)
        self.n_nodes    = len(coords)

    def _generate(self, t_array: np.ndarray, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def as_numpy(self) -> np.ndarray:
        """Return (T, n_nodes) float32 array."""
        return self._generate(self.t_array, self.coords)

    def as_tensor(
        self, device: Union[str, torch.device] = "cpu"
    ) -> Tensor:
        """Return (T, n_nodes) float32 torch Tensor."""
        return torch.tensor(self.as_numpy(), dtype=torch.float32, device=device)

    def as_batch_tensor(
        self,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ) -> Tensor:
        """Return (batch, T, n_nodes) tensor (same stimulus tiled over batch)."""
        t = self.as_tensor(device=device)              # (T, N)
        return t.unsqueeze(0).expand(batch_size, -1, -1)  # (B, T, N)


# ---------------------------------------------------------------------------
# 1. Flash stimulus
# ---------------------------------------------------------------------------

class FlashStimulus(BaseStimulus):
    """
    Full-field luminance step (on-flash or off-flash).

    Parameters
    ----------
    flash_on : float
        Time (ms) when flash turns on.
    flash_off : float
        Time (ms) when flash turns off.
    amplitude : float
        Luminance during flash (0–1). Use negative for off-flash.
    """

    def __init__(
        self,
        coords: np.ndarray,
        flash_on:  float = 100.0,
        flash_off: float = 200.0,
        amplitude: float = 1.0,
        **kwargs,
    ):
        super().__init__(coords, **kwargs)
        self.flash_on  = flash_on
        self.flash_off = flash_off
        self.amplitude = amplitude

    def _generate(self, t_array, coords) -> np.ndarray:
        stim = np.full((len(t_array), len(coords)), self.background, dtype=np.float32)
        mask = (t_array >= self.flash_on) & (t_array < self.flash_off)  # (T,)
        stim[mask, :] = self.amplitude
        return stim


# ---------------------------------------------------------------------------
# 2. Moving bar stimulus
# ---------------------------------------------------------------------------

class MovingBarStimulus(BaseStimulus):
    """
    A bright rectangular bar sweeping across the visual field.

    Parameters
    ----------
    direction_deg : float
        Direction of motion in degrees (0 = rightward, 90 = upward).
    speed : float
        Bar speed in grid-units per ms.
    bar_width : float
        Half-width of the bar in grid units.
    amplitude : float
        Luminance of the bar (0–1).
    """

    def __init__(
        self,
        coords: np.ndarray,
        direction_deg: float = 0.0,
        speed:         float = 0.05,
        bar_width:     float = 1.5,
        amplitude:     float = 1.0,
        **kwargs,
    ):
        super().__init__(coords, **kwargs)
        self.direction_deg = direction_deg
        self.speed         = speed
        self.bar_width     = bar_width
        self.amplitude     = amplitude

    def _generate(self, t_array, coords) -> np.ndarray:
        rad  = np.deg2rad(self.direction_deg)
        d    = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)  # unit vector
        proj = coords @ d                     # (n_nodes,) projection along motion axis

        # bar centre starts just off the min edge and sweeps to max edge
        p_min = proj.min() - self.bar_width
        p_max = proj.max() + self.bar_width
        centre = p_min + (p_max - p_min) * (t_array - t_array[0]) / (t_array[-1] - t_array[0] + 1e-8)
        # centre shape: (T,);  proj shape: (n_nodes,)
        dist   = np.abs(proj[None, :] - centre[:, None])   # (T, n_nodes)
        stim   = np.where(dist <= self.bar_width, self.amplitude, self.background)
        return stim.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Moving edge stimulus
# ---------------------------------------------------------------------------

class MovingEdgeStimulus(BaseStimulus):
    """
    A luminance step-edge moving across the visual field.

    Nodes on one side of the edge are at `amplitude`, the other at `background`.
    """

    def __init__(
        self,
        coords: np.ndarray,
        direction_deg: float = 0.0,
        speed:         float = 0.05,
        amplitude:     float = 1.0,
        **kwargs,
    ):
        super().__init__(coords, **kwargs)
        self.direction_deg = direction_deg
        self.speed         = speed
        self.amplitude     = amplitude

    def _generate(self, t_array, coords) -> np.ndarray:
        rad  = np.deg2rad(self.direction_deg)
        d    = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
        proj = coords @ d                        # (n_nodes,)

        p_min = proj.min()
        p_max = proj.max()
        edge_pos = p_min + self.speed * (t_array - t_array[0])  # (T,)
        # Nodes with projection < edge_pos are in the bright half
        stim = np.where(
            proj[None, :] < edge_pos[:, None],
            self.amplitude,
            self.background,
        )
        return stim.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Sinusoidal grating
# ---------------------------------------------------------------------------

class SinusoidalGrating(BaseStimulus):
    """
    Spatiotemporal sinusoidal grating.

    Parameters
    ----------
    spatial_freq : float
        Cycles per grid-unit.
    temporal_freq : float
        Cycles per ms.
    orientation_deg : float
        Grating orientation in degrees.
    contrast : float
        Contrast (0–1).
    """

    def __init__(
        self,
        coords: np.ndarray,
        spatial_freq:    float = 0.1,
        temporal_freq:   float = 0.01,
        orientation_deg: float = 0.0,
        contrast:        float = 1.0,
        **kwargs,
    ):
        super().__init__(coords, **kwargs)
        self.spatial_freq    = spatial_freq
        self.temporal_freq   = temporal_freq
        self.orientation_deg = orientation_deg
        self.contrast        = contrast

    def _generate(self, t_array, coords) -> np.ndarray:
        rad  = np.deg2rad(self.orientation_deg)
        d    = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
        proj = coords @ d                     # (n_nodes,)
        # Phase = spatial + temporal component
        phase = (
            2 * np.pi * self.spatial_freq  * proj[None, :]       # (1, N)
            - 2 * np.pi * self.temporal_freq * t_array[:, None]  # (T, 1)
        )  # (T, N)
        stim = 0.5 + 0.5 * self.contrast * np.sin(phase)
        return stim.astype(np.float32)


# ---------------------------------------------------------------------------
# Utility: build a stimulus tensor from a sequence description
# ---------------------------------------------------------------------------

def build_stimulus_tensor(
    stimulus_type: str,
    coords: np.ndarray,
    batch_size: int = 1,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
) -> Tensor:
    """
    Convenience factory to build a (batch, T, n_nodes) stimulus tensor.

    Parameters
    ----------
    stimulus_type : str
        One of ``'flash'``, ``'moving_bar'``, ``'moving_edge'``, ``'grating'``.
    coords : np.ndarray  (n_nodes, 2)
        Hex-grid or arbitrary 2-D coordinates of input neurons.
    batch_size : int
        Number of stimulus copies in the batch dimension.
    device : str or torch.device
    **kwargs :
        Forwarded to the stimulus class constructor.

    Returns
    -------
    Tensor  (batch, T, n_nodes)
    """
    _registry = {
        "flash":       FlashStimulus,
        "moving_bar":  MovingBarStimulus,
        "moving_edge": MovingEdgeStimulus,
        "grating":     SinusoidalGrating,
    }
    if stimulus_type not in _registry:
        raise ValueError(f"Unknown stimulus type '{stimulus_type}'. "
                         f"Available: {list(_registry)}.")
    stim_obj = _registry[stimulus_type](coords=coords, **kwargs)
    return stim_obj.as_batch_tensor(batch_size=batch_size, device=device)
