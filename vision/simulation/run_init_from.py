#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same as ``run.py``, but always init from ``RUN_PATH`` (or ``--init-from``).

Usage (from ``simulation/``, project ``.venv``):

    ../.venv/bin/python run_init_from.py --model hp_lp --n-iter 30 --lrs 0.1
"""
from __future__ import annotations

from const_default import (
    RUN_PATH,
)

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import import_bootstrap  # noqa: F401
from const_default import RUN_PATH

# ``from run import`` is ambiguous: run.py and run.slurm share stem ``run``.
_spec = importlib.util.spec_from_file_location(
    "_run_module", os.path.join(HERE, "run.py"),
)
_run = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_run)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print(f"cli: {' '.join(argv)}")
    parser = _run.build_run_argparser()
    args = parser.parse_args(argv)
    try:
        kwargs = _run.resolve_run_kwargs(args, script_stem="run_init_from")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    kwargs["init_from"] = args.init_from or RUN_PATH
    _run.run_train_and_plot(**kwargs)


if __name__ == "__main__":
    main()
