"""Single source for locating the FAFB connectome folder.

The FAFB package lives under ``simulation/../connectome/FAFBv783``.
Importing this module ensures ``vision/import_bootstrap`` is installed so
simulation can ``import build_hex`` / ``path``. ``network.construction``
(``hex_in_cost_extent``), the task paradigms, and the training/plot layers
import this at module load.

Disk path is ``simulation/2_network/1_path.py`` (sorted ``N_name``
paths); logical import remains ``network.path``.
"""

from __future__ import annotations

from pathlib import Path

import import_bootstrap  # noqa: F401 — vision logical-import finder + sys.path

# This file lives at simulation/<N>_network/<N>_path.py -> three
# parents up is ``vision`` which holds ``connectome/FAFBv783``.
FAFB_DIR = Path(__file__).resolve().parent.parent.parent / "connectome" / "FAFBv783"
