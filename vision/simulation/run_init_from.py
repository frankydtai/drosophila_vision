#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train from ``init_from`` (defaults to ``RUN_PATH`` when unset).

Usage (from ``simulation/``, project ``.venv``):

    ../.venv/bin/python run_init_from.py
    ../.venv/bin/python run_init_from.py n_iter=30
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import import_bootstrap  # noqa: F401

import hydra

from config import RUN_PATH, resolve_run_kwargs
from run import run_train_and_plot


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(hydra_config) -> None:
    try:
        kwargs = resolve_run_kwargs(hydra_config, script_stem="run_init_from")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not kwargs.get("init_from"):
        kwargs["init_from"] = RUN_PATH
    run_train_and_plot(**kwargs)


if __name__ == "__main__":
    main()
