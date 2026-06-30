"""Single source for locating the FAFB connectome folder.

The FAFB package lives one level up under ``Connectome/FAFB v783`` (the path has
a space, so it is not importable as a normal package). Importing this module adds
that folder to ``sys.path`` so SimulationCode can ``import column_mapper`` / ``fafb_io``
/ ``Medulla_Library``. Defined once here so the path string is never duplicated.
"""

from __future__ import annotations

import sys
from pathlib import Path

FAFB_DIR = Path(__file__).resolve().parent.parent / "Connectome" / "FAFB v783"

if str(FAFB_DIR) not in sys.path:
    sys.path.insert(0, str(FAFB_DIR))
