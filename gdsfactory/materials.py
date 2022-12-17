"""Register materials."""
from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy as np

MaterialSpec = Union[str, float, Tuple[float, float], Callable]

material_name_to_meep = {
    "si": "Si",
    "sin": "Si3N4_NIR",
    "sio2": "SiO2",
}


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


materials_index = dict(si=si, sio2=sio2, sin=sin)
