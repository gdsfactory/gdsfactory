"""gdsfactory tidy3d plugin.

[tidy3D is a fast GPU based commercial FDTD
solver](https://simulation.cloud/)

"""

import tidy3d as td

from gdsfactory.config import logger
from gdsfactory.simulation.gtidy3d import materials, modes, utils
from gdsfactory.simulation.gtidy3d.get_results import get_results
from gdsfactory.simulation.gtidy3d.get_simulation import (
    get_simulation,
    plot_simulation,
    plot_simulation_xz,
    plot_simulation_yz,
)
from gdsfactory.simulation.gtidy3d.get_simulation_grating_coupler import (
    get_simulation_grating_coupler,
)
from gdsfactory.simulation.gtidy3d.write_sparameters import (
    write_sparameters,
    write_sparameters_1x1,
    write_sparameters_batch,
    write_sparameters_batch_1x1,
    write_sparameters_crossing,
)
from gdsfactory.simulation.gtidy3d.write_sparameters_grating_coupler import (
    write_sparameters_grating_coupler,
    write_sparameters_grating_coupler_batch,
)

__version__ = "0.0.2"
__all__ = [
    "plot_simulation",
    "plot_simulation_xz",
    "plot_simulation_yz",
    "get_simulation",
    "get_simulation_grating_coupler",
    "get_results",
    "materials",
    "modes",
    "utils",
    "write_sparameters",
    "write_sparameters_crossing",
    "write_sparameters_1x1",
    "write_sparameters_batch",
    "write_sparameters_batch_1x1",
    "write_sparameters_grating_coupler",
    "write_sparameters_grating_coupler_batch",
]

logger.info(f"Tidy3d {td.__version__!r} installed at {td.__path__!r}")
