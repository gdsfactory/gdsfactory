"""gdsfactory tidy3d plugin

[tidy3D is a fast GPU based commercial FDTD solver](https://simulation.cloud/)
"""

from gdsfactory.simulation.gtidy3d import materials
from gdsfactory.simulation.gtidy3d.get_results import get_results
from gdsfactory.simulation.gtidy3d.get_simulation import (
    get_simulation,
    plot_simulation,
    plot_simulation_xz,
    plot_simulation_yz,
)
from gdsfactory.simulation.gtidy3d.write_sparameters import (
    write_sparameters,
    write_sparameters_1x1,
    write_sparameters_crossing,
)

__version__ = "0.0.2"
__all__ = [
    "plot_simulation",
    "plot_simulation_xz",
    "plot_simulation_yz",
    "get_simulation",
    "get_results",
    "materials",
    "write_sparameters",
    "write_sparameters_crossing",
    "write_sparameters_1x1",
]
