import numpy as np
from gdsfactory.technology.simulation_settings import SimulationSettingsLumericalFdtd

SIMULATION_SETTINGS_LUMERICAL_FDTD = SimulationSettingsLumericalFdtd()


# default materials
def si(wav: np.ndarray) -> np.ndarray:
    """Silicon crystalline."""
    from gdsfactory.simulation.gtidy3d.materials import si

    return si(wav)


def sio2(wav: np.ndarray) -> np.ndarray:
    """Silicon oxide."""
    from gdsfactory.simulation.gtidy3d.materials import sio2

    return sio2(wav)


def sin(wav: np.ndarray) -> np.ndarray:
    """Silicon Nitride."""
    from gdsfactory.simulation.gtidy3d.materials import sin

    return sin(wav)


materials_index = {"si": si, "sio2": sio2, "sin": sin}
