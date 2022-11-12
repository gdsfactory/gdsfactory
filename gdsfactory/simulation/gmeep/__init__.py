import meep as mp

from gdsfactory.config import logger
from gdsfactory.simulation import plot, port_symmetries
from gdsfactory.simulation.get_sparameters_path import get_sparameters_data_meep
from gdsfactory.simulation.gmeep.get_simulation import get_simulation
from gdsfactory.simulation.gmeep.meep_adjoint_optimization import (
    get_meep_adjoint_optimizer,
    run_meep_adjoint_optimizer,
)
from gdsfactory.simulation.gmeep.write_sparameters_grating import (
    write_sparameters_grating,
    write_sparameters_grating_batch,
    write_sparameters_grating_mpi,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep import (
    write_sparameters_meep,
    write_sparameters_meep_1x1,
    write_sparameters_meep_1x1_bend90,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep_batch import (
    write_sparameters_meep_batch,
    write_sparameters_meep_batch_1x1,
    write_sparameters_meep_batch_1x1_bend90,
)
from gdsfactory.simulation.gmeep.write_sparameters_meep_mpi import (
    write_sparameters_meep_mpi,
    write_sparameters_meep_mpi_1x1,
    write_sparameters_meep_mpi_1x1_bend90,
)

logger.info(f"Meep {mp.__version__!r} installed at {mp.__path__!r}")

__all__ = [
    "get_meep_adjoint_optimizer",
    "get_simulation",
    "get_sparameters_data_meep",
    "run_meep_adjoint_optimizer",
    "write_sparameters_meep",
    "write_sparameters_meep_1x1",
    "write_sparameters_meep_1x1_bend90",
    "write_sparameters_meep_mpi",
    "write_sparameters_meep_mpi_1x1",
    "write_sparameters_meep_mpi_1x1_bend90",
    "write_sparameters_meep_batch",
    "write_sparameters_meep_batch_1x1",
    "write_sparameters_meep_batch_1x1_bend90",
    "write_sparameters_grating",
    "write_sparameters_grating_mpi",
    "write_sparameters_grating_batch",
    "plot",
    "port_symmetries",
]
__version__ = "0.0.3"
