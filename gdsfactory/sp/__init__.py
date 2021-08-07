"""Work with SParameters.
"""

from gdsfactory.sp.get_sparameters_path import get_sparameters_path
from gdsfactory.sp.plot import plot
from gdsfactory.sp.read import read_sparameters_component, read_sparameters_lumerical
from gdsfactory.sp.write import write

__all__ = [
    "get_sparameters_path",
    "plot",
    "read_sparameters_component",
    "read_sparameters_lumerical",
    "write",
]
