# -*- coding: utf-8 -*-
"""T4 / T5 moving-bar preference: PD/ND x PC/NC and fig1_ci trace keys.

Rules match ``t4_t5_preference.md``. Orthogonal motion (``—`` in the tables)
returns ``None`` so those (stimulus, subtype) pairs are skipped in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from visual_stimulus.moving_bar_stimulus import GRUNTMAN_WIDTHS_DEG, MovingBarSpec

READOUT_SUBTYPES: Tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d",
    "T5a", "T5b", "T5c", "T5d",
)

_HORIZONTAL = frozenset({"right", "left"})
_VERTICAL = frozenset({"up", "down"})
# Plot / stimulus iteration order (matches gruntman_moving_bar_specs).
_HORIZONTAL_AXIS: Tuple[str, ...] = ("right", "left")
_VERTICAL_AXIS: Tuple[str, ...] = ("up", "down")
_OPPOSITE = {"right": "left", "left": "right", "up": "down", "down": "up"}

# Subtype PD on the **right** eye (anterior->posterior = right; c/d = up/down).
_SUBTYPE_PD_RIGHT = {"a": "right", "b": "left", "c": "up", "d": "down"}


@dataclass(frozen=True)
class MotionPreference:
    """PD/ND from motion; PC/NC from contrast + pathway (T4 vs T5)."""

    pd_nd: str  # "PD" | "ND"
    pc_nc: str  # "PC" | "NC"


def normalize_side(side: str) -> str:
    s = str(side).strip().lower()
    if s in ("r", "right"):
        return "right"
    if s in ("l", "left"):
        return "left"
    raise ValueError(f"unknown eye side {side!r}")


def width_tag(width_deg: float) -> str:
    return "w1" if float(width_deg) <= 3.0 else "w4"


def pd_direction(side: str, subtype: str) -> str:
    """Preferred-direction motion for ``subtype`` on ``side`` (right or left eye)."""
    side = normalize_side(side)
    letter = subtype[-1]
    if letter not in _SUBTYPE_PD_RIGHT:
        raise ValueError(f"unknown subtype {subtype!r}")
    d = _SUBTYPE_PD_RIGHT[letter]
    if side == "left" and d in _HORIZONTAL:
        return _OPPOSITE[d]
    return d


def _axis_directions(subtype: str) -> Tuple[str, ...]:
    return _HORIZONTAL_AXIS if subtype[-1] in "ab" else _VERTICAL_AXIS


def motion_preference(
    side: str,
    subtype: str,
    direction: str,
    contrast: str,
) -> Optional[MotionPreference]:
    """Map one cardinal stimulus to PD/ND + PC/NC for a T4/T5 subtype.

    Returns ``None`` when motion is orthogonal to the subtype PD axis (table ``—``).
    """
    if subtype not in READOUT_SUBTYPES:
        raise ValueError(f"unknown subtype {subtype!r}")
    direction = str(direction).strip().lower()
    contrast = str(contrast).strip().lower()
    if direction not in _HORIZONTAL | _VERTICAL:
        raise ValueError(f"unknown direction {direction!r}")
    if contrast not in ("bright", "dark"):
        raise ValueError(f"unknown contrast {contrast!r}")

    if direction not in _axis_directions(subtype):
        return None

    pd_dir = pd_direction(side, subtype)
    if direction == pd_dir:
        pd_nd = "PD"
    elif direction == _OPPOSITE[pd_dir]:
        pd_nd = "ND"
    else:
        return None

    pathway = subtype[:2]
    if pathway == "T4":
        pc_nc = "PC" if contrast == "bright" else "NC"
    elif pathway == "T5":
        pc_nc = "PC" if contrast == "dark" else "NC"
    else:
        raise ValueError(f"unknown pathway in {subtype!r}")

    return MotionPreference(pd_nd=pd_nd, pc_nc=pc_nc)


def fig1_trace_key(pathway: str, pc_nc: str, width: str, pd_nd: str) -> str:
    """Key in ``fig1_ci_digitized.npz`` (e.g. ``T4_PC_w1_PD``)."""
    return f"{pathway}_{pc_nc}_{width}_{pd_nd}"


def fig1_key_for_stimulus(
    side: str,
    subtype: str,
    spec: Union[MovingBarSpec, str],
    contrast: Optional[str] = None,
    width_deg: Optional[float] = None,
) -> Optional[str]:
    """fig1 trace id for ``(side, subtype, stimulus)``, or ``None`` if orthogonal."""
    if isinstance(spec, MovingBarSpec):
        direction, contrast, width_deg = spec.direction, spec.contrast, spec.width_deg
    else:
        direction = str(spec)
        if contrast is None or width_deg is None:
            raise ValueError("contrast and width_deg required when spec is not MovingBarSpec")
    pref = motion_preference(side, subtype, direction, contrast)
    if pref is None:
        return None
    return fig1_trace_key(subtype[:2], pref.pc_nc, width_tag(width_deg), pref.pd_nd)


def active_stimuli_for_subtype(side: str, subtype: str) -> Sequence[Tuple[str, str, str]]:
    """Non-orthogonal (direction, contrast, width_tag) triples for one subtype (8 total)."""
    out = []
    for direction in _axis_directions(subtype):
        for contrast in ("bright", "dark"):
            for w in GRUNTMAN_WIDTHS_DEG:
                if motion_preference(side, subtype, direction, contrast) is not None:
                    out.append((direction, contrast, width_tag(w)))
    return out
