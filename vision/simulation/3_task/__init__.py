"""Task layer: per-paradigm input (geometry + sti current), gt, and pack.

Each paradigm package exposes ``input`` and ``gt``: geometry + drive in
``input``; gt traces in ``gt``. Spot also has ``pack`` (GT↔network cost
binding). Leak / ir scale live in ``neuron.param``.
"""
