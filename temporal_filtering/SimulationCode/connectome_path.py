"""Single source for locating the FAFB connectome folder.

The FAFB package lives one level up under ``Connectome/FAFBv783``. Importing
this module adds that folder to ``sys.path`` so SimulationCode can
``import column_mapper`` / ``connectome_io``. ``network.stimulus``,
``network.tiling``, and ``visual_stimulus.moving_bar_stimulus`` import this
at module load.
"""

from __future__ import annotations

import sys
from pathlib import Path

FAFB_DIR = Path(__file__).resolve().parent.parent / "Connectome" / "FAFBv783"

if str(FAFB_DIR) not in sys.path:
    sys.path.insert(0, str(FAFB_DIR))
