"""Task layer: per-task-family input (geometry + sti current), gt, and pack.

``spread`` / ``spot`` / ``mbar`` packages expose sti geometry + drive,
gt traces, and (where applicable) pack builders. Shared timing / IR live in
``spread``; spot imports from spread. Session scalars live in ``config.MODEL``.
"""
