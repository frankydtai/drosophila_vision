"""FAFB connectome on ``sys.path`` via ``import_bootstrap``.

Importing this module (``import network.path`` / ``from network import path``)
installs the vision logical-import finder so flat FAFBv783 modules
(``build_hex``, …) resolve. Disk file: ``simulation/2_network/1_path.py``;
logical import remains ``network.path``.
"""

from __future__ import annotations

import import_bootstrap  # noqa: F401 — vision logical-import finder + sys.path
