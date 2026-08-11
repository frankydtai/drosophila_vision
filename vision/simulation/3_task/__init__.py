"""Task layer: per-paradigm input (geometry + sti current) and gt.

Each paradigm package exposes ``input`` and ``gt`` submodules: geometry +
drive live in ``input``; gt traces + cost readout live in ``gt``.
Leak / ImpR scale live in ``neuron.params``.
"""
