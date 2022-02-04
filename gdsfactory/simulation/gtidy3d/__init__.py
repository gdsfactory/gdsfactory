"""gdsfactory tidy3d plugin

[tidy3D is a FDTD GPU commercial](https://simulation.cloud/)
"""

from gdsfactory.simulation.gtidy3d import materials
from gdsfactory.simulation.gtidy3d.get_results import get_results
from gdsfactory.simulation.gtidy3d.get_simulation import get_simulation, plot_simulation
from gdsfactory.simulation.gtidy3d.get_sparameters import get_sparameters

__version__ = "0.0.1"
__all__ = [
    "plot_simulation",
    "get_sparameters",
    "get_simulation",
    "get_results",
    "materials",
]
