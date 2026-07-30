"""Task layer: per-paradigm input (geometry + PR current) and data (target).

Each paradigm package exposes ``input`` and ``data`` submodules: geometry +
drive live in ``input``; target traces + cost readout layout live in ``data``.
Leak / ImpR scale live in ``neuron_model.params``.
"""
