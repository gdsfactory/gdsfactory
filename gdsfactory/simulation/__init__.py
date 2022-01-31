"""gdsfactory interface to simulations."""

from gdsfactory.simulation import plot
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_data_lumerical,
    get_sparameters_data_meep,
    get_sparameters_path_lumerical,
    get_sparameters_path_meep,
)

__all__ = [
    "plot",
    "get_sparameters_path_meep",
    "get_sparameters_path_lumerical",
    "get_sparameters_data_meep",
    "get_sparameters_data_lumerical",
]
