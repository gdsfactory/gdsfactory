"""gdsfactory interface to simulations."""

from gdsfactory.simulation import plot
from gdsfactory.simulation.get_effective_indices import get_effective_indices
from gdsfactory.simulation.get_sparameters_path import (
    get_sparameters_data_lumerical,
    get_sparameters_data_meep,
    get_sparameters_data_tidy3d,
    get_sparameters_path_lumerical,
    get_sparameters_path_meep,
    get_sparameters_path_tidy3d,
)

__all__ = [
    "plot",
    "get_sparameters_path_meep",
    "get_sparameters_path_lumerical",
    "get_sparameters_path_tidy3d",
    "get_sparameters_data_meep",
    "get_sparameters_data_lumerical",
    "get_sparameters_data_tidy3d",
    "get_effective_indices",
]
