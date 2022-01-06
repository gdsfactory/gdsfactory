import meep as mp

from gdsfactory.config import logger
from gdsfactory.simulation.gmeep.add_monitors import add_monitors
from gdsfactory.simulation.gmeep.get_simulation import get_simulation
from gdsfactory.simulation.gmeep.get_transmission_2ports import (
    get_transmission_2ports,
    plot2D,
    plot3D,
)
from gdsfactory.simulation.gmeep.plot_xsection import plot_xsection
from gdsfactory.simulation.gmeep.write_sparameters_meep import write_sparameters_meep
from gdsfactory.simulation.gmeep.write_sparameters_meep_1x1 import (
    write_sparameters_meep_1x1,
)

logger.info(f"Found Meep {mp.__version__!r} installed at {mp.__path__!r}")

__all__ = [
    "add_monitors",
    "get_simulation",
    "write_sparameters_meep",
    "write_sparameters_meep_1x1",
    "get_transmission_2ports",
    "plot2D",
    "plot3D",
    "plot_xsection",
]
__version__ = "0.0.2"
