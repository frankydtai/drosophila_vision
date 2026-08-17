"""Task layer: per-task-family input (geometry + sti current), gt, and pack.

Each ``spot`` / ``moving_bar`` package exposes ``input`` and ``gt``: geometry + drive in
``input``; gt traces in ``gt``. Spot also has ``pack`` (GT↔network cost
binding). Session scalars live in ``const_default.MODEL``.
"""
