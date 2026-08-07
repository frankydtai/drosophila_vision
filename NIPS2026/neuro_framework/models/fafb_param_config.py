"""Configuration loaders for FAFB ion/synapse parameter priors and fallback rules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd

DEFAULT_ION = {
    "gNa": 0.003,
    "gK": 0.002,
    "gLeak": 0.0015,
    "eLeak": -50.0,
    "eNa": 50.0,
    "eK": -77.0,
}

DEFAULT_CELL_TYPE_ALIAS = {
    "R1-R6": "R1-6",
}

_DEFAULT_RULE_PATH = Path(__file__).parents[1] / "data" / "fafb_type_fallback_rules.csv"
_DEFAULT_NT_PRIOR_PATH = Path(__file__).parents[1] / "data" / "fafb_nt_ion_priors.csv"


def load_nt_ion_priors(path: Union[str, Path] = _DEFAULT_NT_PRIOR_PATH) -> Dict[str, Dict]:
    df = pd.read_csv(path)
    priors = {}
    for _, r in df.iterrows():
        priors[str(r["nt_type"]).strip().lower()] = {
            "gNa": float(r["gNa"]),
            "gK": float(r["gK"]),
            "gLeak": float(r["gLeak"]),
            "eNa": float(r["eNa"]),
            "eK": float(r["eK"]),
            "eLeak": float(r["eLeak"]),
        }
    if "unknown" not in priors:
        priors["unknown"] = dict(DEFAULT_ION)
    return priors


def load_type_fallback_rules(path: Union[str, Path] = _DEFAULT_RULE_PATH) -> Tuple[Dict[str, str], Dict[str, str]]:
    df = pd.read_csv(path)
    aliases = dict(DEFAULT_CELL_TYPE_ALIAS)
    fallback = {}
    for _, r in df.iterrows():
        src = str(r["source_type"]).strip()
        dst = str(r["target_type"]).strip()
        kind = str(r.get("rule_kind", "fallback")).strip().lower()
        if kind == "alias":
            aliases[src] = dst
        else:
            fallback[src] = dst
    return aliases, fallback


__all__ = [
    "DEFAULT_ION",
    "DEFAULT_CELL_TYPE_ALIAS",
    "load_nt_ion_priors",
    "load_type_fallback_rules",
]
