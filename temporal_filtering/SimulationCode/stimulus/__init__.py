"""Stimulus layer: per-paradigm input (geometry + PR current) and data (target).

Each paradigm package exposes exactly ``input`` and ``data`` submodules:
geometry + drive live in ``input``; target traces + cost readout layout live in
``data``. Shared currents / cell list live in :mod:`stimulus.constants`.
"""
