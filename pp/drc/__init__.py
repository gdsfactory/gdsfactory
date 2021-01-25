"""Basic checks."""

from pp.drc.check_exclusion import check_exclusion
from pp.drc.check_inclusion import check_inclusion
from pp.drc.check_space import check_space
from pp.drc.check_width import check_width
from pp.drc.density import compute_area
from pp.drc.snap_to_grid import (
    assert_on_1nm_grid,
    on_1nm_grid,
    on_2nm_grid,
    on_grid,
    snap_to_1nm_grid,
    snap_to_2nm_grid,
    snap_to_5nm_grid,
    snap_to_grid,
)

__all__ = [
    "check_space",
    "check_width",
    "check_exclusion",
    "check_inclusion",
    "compute_area",
    "on_grid",
    "on_1nm_grid",
    "on_2nm_grid",
    "assert_on_1nm_grid",
    "assert_on_2nm_grid",
    "snap_to_grid",
    "snap_to_1nm_grid",
    "snap_to_2nm_grid",
    "snap_to_5nm_grid",
]
