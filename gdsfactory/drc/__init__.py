"""Basic checks."""

from gdsfactory.drc.check_duplicate_cells import check_duplicate_cells
from gdsfactory.drc.check_exclusion import check_exclusion
from gdsfactory.drc.check_inclusion import check_inclusion
from gdsfactory.drc.check_space import check_space
from gdsfactory.drc.check_width import check_width
from gdsfactory.drc.density import compute_area

__all__ = [
    "check_space",
    "check_width",
    "check_exclusion",
    "check_inclusion",
    "check_duplicate_cells",
    "compute_area",
]
