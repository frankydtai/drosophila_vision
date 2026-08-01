"""Task layer: per-paradigm input (geometry + PR current) and data (gt).

Each paradigm package exposes ``input`` and ``data`` submodules: geometry +
drive live in ``input``; gt traces + cost readout live in ``data``.
Leak / ImpR scale live in ``neuron.params``.
"""
