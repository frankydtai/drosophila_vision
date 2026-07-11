"""Regression: t4_t5_preference matches t4_t5_preference.md tables."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from t4_t5_preference import READOUT_SUBTYPES, active_stimuli_for_subtype, fig1_key_for_stimulus, motion_preference


# Parsed from t4_t5_preference.md (right / left eye tables).
_RIGHT = {
  ("right", "bright"): {"T4a": ("PD", "PC"), "T4b": ("ND", "PC"), "T5a": ("PD", "NC"), "T5b": ("ND", "NC")},
  ("right", "dark"): {"T4a": ("PD", "NC"), "T4b": ("ND", "NC"), "T5a": ("PD", "PC"), "T5b": ("ND", "PC")},
  ("left", "bright"): {"T4a": ("ND", "PC"), "T4b": ("PD", "PC"), "T5a": ("ND", "NC"), "T5b": ("PD", "NC")},
  ("left", "dark"): {"T4a": ("ND", "NC"), "T4b": ("PD", "NC"), "T5a": ("ND", "PC"), "T5b": ("PD", "PC")},
  ("up", "bright"): {"T4c": ("PD", "PC"), "T4d": ("ND", "PC"), "T5c": ("PD", "NC"), "T5d": ("ND", "NC")},
  ("up", "dark"): {"T4c": ("PD", "NC"), "T4d": ("ND", "NC"), "T5c": ("PD", "PC"), "T5d": ("ND", "PC")},
  ("down", "bright"): {"T4c": ("ND", "PC"), "T4d": ("PD", "PC"), "T5c": ("ND", "NC"), "T5d": ("PD", "NC")},
  ("down", "dark"): {"T4c": ("ND", "NC"), "T4d": ("PD", "NC"), "T5c": ("ND", "PC"), "T5d": ("PD", "PC")},
}

_LEFT = {
  ("left", "bright"): {"T4a": ("PD", "PC"), "T4b": ("ND", "PC"), "T5a": ("PD", "NC"), "T5b": ("ND", "NC")},
  ("left", "dark"): {"T4a": ("PD", "NC"), "T4b": ("ND", "NC"), "T5a": ("PD", "PC"), "T5b": ("ND", "PC")},
  ("right", "bright"): {"T4a": ("ND", "PC"), "T4b": ("PD", "PC"), "T5a": ("ND", "NC"), "T5b": ("PD", "NC")},
  ("right", "dark"): {"T4a": ("ND", "NC"), "T4b": ("PD", "NC"), "T5a": ("ND", "PC"), "T5b": ("PD", "PC")},
  ("up", "bright"): {"T4c": ("PD", "PC"), "T4d": ("ND", "PC"), "T5c": ("PD", "NC"), "T5d": ("ND", "NC")},
  ("up", "dark"): {"T4c": ("PD", "NC"), "T4d": ("ND", "NC"), "T5c": ("PD", "PC"), "T5d": ("ND", "PC")},
  ("down", "bright"): {"T4c": ("ND", "PC"), "T4d": ("PD", "PC"), "T5c": ("ND", "NC"), "T5d": ("PD", "NC")},
  ("down", "dark"): {"T4c": ("ND", "NC"), "T4d": ("PD", "NC"), "T5c": ("ND", "PC"), "T5d": ("PD", "PC")},
}


def _check_table(side: str, table: dict):
  for subtype in READOUT_SUBTYPES:
    axis = "horizontal" if subtype[-1] in "ab" else "vertical"
    for direction in ("right", "left", "up", "down"):
      for contrast in ("bright", "dark"):
        pref = motion_preference(side, subtype, direction, contrast)
        key = (direction, contrast)
        if axis == "horizontal" and direction in ("up", "down"):
          assert pref is None, (side, subtype, key)
          continue
        if axis == "vertical" and direction in ("right", "left"):
          assert pref is None, (side, subtype, key)
          continue
        exp = table[key][subtype]
        assert pref is not None, (side, subtype, key)
        assert (pref.pd_nd, pref.pc_nc) == exp, (side, subtype, key, pref, exp)


def test_right_eye_table():
  _check_table("right", _RIGHT)


def test_left_eye_table():
  _check_table("left", _LEFT)


def test_fig1_key_example():
  assert fig1_key_for_stimulus("right", "T4a", "right", "bright", 2.25) == "T4_PC_w1_PD"
  assert fig1_key_for_stimulus("right", "T5a", "right", "bright", 9.0) == "T5_NC_w4_PD"
  assert fig1_key_for_stimulus("right", "T4c", "right", "bright", 2.25) is None


def test_active_count():
  for side in ("right", "left"):
    for subtype in READOUT_SUBTYPES:
      assert len(active_stimuli_for_subtype(side, subtype)) == 8


def test_active_stimulus_order():
  bright = [
    f"{d}_{c}_{w}"
    for d, c, w in active_stimuli_for_subtype("right", "T4a")
    if c == "bright"
  ]
  assert bright == [
    "right_bright_w1", "right_bright_w4",
    "left_bright_w1", "left_bright_w4",
  ]
