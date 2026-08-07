from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MALE_CNS_SWC_SCALE_TO_UM = 0.008


def find_skeleton_dir(data_dir: Path) -> Path:
    """
    Locate the Male CNS SWC skeleton directory.

    The download often ends up in one of these layouts:
      - <data_dir>/skelton/skeletons-swc            (common typo in folder name)
      - <data_dir>/skeleton/skeletons-swc
      - <data_dir>/skeletons-swc
    """
    candidates = [
        data_dir / "skelton" / "skeletons-swc",
        data_dir / "skeleton" / "skeletons-swc",
        data_dir / "skeletons-swc",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find Male CNS skeletons-swc folder under {data_dir}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


@dataclass(frozen=True)
class MaleCNSSkeletonIndex:
    """
    Resolve SWC file paths by body id.

    Parameters
    ----------
    skeleton_dir:
        Directory that contains SWC files named like '<body_id>.swc'.
    scale_to_um:
        Coordinate/radius scale factor to convert file units to microns.
        For Janelia Male CNS `skeletons-swc`, this is typically 0.008.
    """

    skeleton_dir: Path
    scale_to_um: float = MALE_CNS_SWC_SCALE_TO_UM

    @classmethod
    def from_data_dir(cls, data_dir: Path, *, prefer_um: bool = True) -> "MaleCNSSkeletonIndex":
        """
        Build an index from the dataset root folder.

        If `<data_dir>/.../skeletons-swc-um` exists, we prefer it (units already in μm).
        Otherwise we fall back to `skeletons-swc` and set `scale_to_um=0.008`.
        """
        # Prefer a μm folder if present.
        um_candidates = [
            data_dir / "skelton" / "skeletons-swc-um",
            data_dir / "skeleton" / "skeletons-swc-um",
            data_dir / "skeletons-swc-um",
        ]
        def _has_enough_swc_files(path: Path, *, min_files: int = 10) -> bool:
            if not (path.exists() and path.is_dir()):
                return False
            n = 0
            for entry in path.iterdir():
                if entry.is_file() and entry.suffix.lower() == ".swc":
                    n += 1
                    if n >= min_files:
                        return True
            return False

        if prefer_um:
            for c in um_candidates:
                # Some downloads create an empty or partial `skeletons-swc-um/`.
                # Only prefer it if it looks complete enough.
                if _has_enough_swc_files(c, min_files=10):
                    return cls(skeleton_dir=c, scale_to_um=1.0)

        return cls(skeleton_dir=find_skeleton_dir(data_dir), scale_to_um=MALE_CNS_SWC_SCALE_TO_UM)

    def swc_path(self, body_id: int) -> Path:
        return self.skeleton_dir / f"{int(body_id)}.swc"

    def exists(self, body_id: int) -> bool:
        return self.swc_path(body_id).exists()

    def require(self, body_id: int) -> Path:
        path = self.swc_path(body_id)
        if not path.exists():
            raise FileNotFoundError(f"SWC not found for body_id={int(body_id)}: {path}")
        return path


def swc_path_for_body_id(data_dir: Path, body_id: int, *, prefer_um: bool = True) -> Path:
    """Convenience wrapper around `MaleCNSSkeletonIndex.from_data_dir(...).swc_path(...)`."""
    return MaleCNSSkeletonIndex.from_data_dir(data_dir, prefer_um=prefer_um).swc_path(body_id)
