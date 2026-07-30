"""Single source for locating the FAFB connectome folder.

The FAFB package lives under ``SimulationCode/../Connectome/FAFBv783``.
Importing this module adds that folder to ``sys.path`` so SimulationCode can
``import column_mapper`` / ``connectome_io``. ``network.layout`` (shared
``column_in_cost_extent``), the task paradigms, and the training/plot
layers import this at module load.

Disk path is ``SimulationCode/2_network/1_path.py`` (sorted ``N_name``
layout); logical import remains ``network.path``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This file lives at SimulationCode/<N>_network/<N>_bootstrap.py -> three
# parents up is ``temporal_filtering`` which holds ``Connectome/FAFBv783``.
FAFB_DIR = Path(__file__).resolve().parent.parent.parent / "Connectome" / "FAFBv783"

if str(FAFB_DIR) not in sys.path:
    sys.path.insert(0, str(FAFB_DIR))

