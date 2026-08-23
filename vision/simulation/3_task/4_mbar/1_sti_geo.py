"""Moving-bar sti geometry re-exports from :mod:`task.sbar.sti_geo`."""
from task.sbar.sti_geo import (
    BAR_RADIUS,
    Hex,
    StiHex,
    bar_bound0_bar_bound1s,
    _fill,
    bar_bounds,
    sti_hexes_at_xy,
    i_sti_nodes_from_hexes,
    node_us_vs,
    sti_hexes,
    view_bounds,
)

GRUNTMAN_WS_DEG = (2.25, 9.0)
GRUNTMAN_directions = ("right", "left", "up", "down")
