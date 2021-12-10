"""
[tidy3D is a FDTD web based software](https://simulation.cloud/)

"""

from gdsfactory.simulation.tidy3d import materials
from gdsfactory.simulation.tidy3d.get_coupling import get_coupling
from gdsfactory.simulation.tidy3d.get_simulation import get_simulation, plot_simulation
from gdsfactory.simulation.tidy3d.get_sparameters import get_sparameters
from gdsfactory.simulation.tidy3d.run_simulation import run_simulation

__version__ = "0.0.1"
__all__ = [
    "plot_simulation",
    "get_coupling",
    "get_sparameters",
    "get_simulation",
    "run_simulation",
    "materials",
]
