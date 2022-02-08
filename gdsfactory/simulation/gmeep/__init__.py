import meep as mp

from gdsfactory.config import logger
from gdsfactory.simulation import plot, port_symmetries
from gdsfactory.simulation.get_sparameters_path import get_sparameters_data_meep
from gdsfactory.simulation.gmeep.get_simulation import get_simulation
from gdsfactory.simulation.gmeep.write_sparameters_meep import (
    write_sparameters_meep,
    write_sparameters_meep_lr,
    write_sparameters_meep_lt,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep_batch import (
    write_sparameters_meep_batch,
    write_sparameters_meep_batch_lr,
    write_sparameters_meep_batch_lt,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep_mpi import (
    write_sparameters_meep_mpi,
    write_sparameters_meep_mpi_lr,
    write_sparameters_meep_mpi_lt,
)

logger.info(f"Meep {mp.__version__!r} installed at {mp.__path__!r}")

__all__ = [
    "get_simulation",
    "get_sparameters_data_meep",
    "write_sparameters_meep",
    "write_sparameters_meep_lr",
    "write_sparameters_meep_lt",
    "write_sparameters_meep_mpi",
    "write_sparameters_meep_mpi_lr",
    "write_sparameters_meep_mpi_lt",
    "write_sparameters_meep_batch",
    "write_sparameters_meep_batch_lr",
    "write_sparameters_meep_batch_lt",
    "plot",
    "port_symmetries",
]
__version__ = "0.0.3"
