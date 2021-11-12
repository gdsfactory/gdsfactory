from functools import partial

import meep as mp
import numpy as np

from gdsfactory.simulation.gmpb.find_modes import find_modes
from gdsfactory.simulation.gmpb.get_mode_solver_rib import get_mode_solver_rib
from gdsfactory.simulation.gmpb.types import WidthSweep


def find_neff_vs_width(
    w1: float = 0.2,
    w2: float = 1.0,
    steps: int = 12,
    mode_solver=get_mode_solver_rib,
    nmodes: int = 4,
    wavelength: float = 1.55,
    parity=mp.NO_PARITY,
    **kwargs
):
    width = np.linspace(w1, w2, steps)

    neff = {}

    for mode_number in range(1, nmodes + 1):
        neff[mode_number] = []

    for wg_width in width:
        mode_solver = partial(mode_solver, wg_width=wg_width, nmodes=nmodes, **kwargs)
        modes = find_modes(mode_solver, wavelength=wavelength, parity=parity)
        for mode_number in range(1, nmodes + 1):
            mode = modes[mode_number]
            neff[mode_number].append(mode.neff)

    return WidthSweep(width=list(width), neff=neff)


if __name__ == "__main__":
    # s = find_neff_vs_width()

    w1: float = 0.5
    w2: float = 1.0
    steps: int = 3
    mode_solver = get_mode_solver_rib
    nmodes: int = 4
    wavelength: float = 1.55
    parity = mp.NO_PARITY
    width = np.linspace(w1, w2, steps)

    neff = {}

    for mode_number in range(1, nmodes + 1):
        neff[mode_number] = []

    for wg_width in width:
        mode_solver = partial(mode_solver, wg_width=wg_width, nmodes=nmodes)
        modes = find_modes(mode_solver, wavelength=wavelength, parity=parity)
        for mode_number in range(1, nmodes + 1):
            mode = modes[mode_number]
            neff[mode_number].append(mode.neff)

    s = WidthSweep(width=list(width), neff=neff)
