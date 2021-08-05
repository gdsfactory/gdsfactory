"""Sweep neff over wavelength and returns group index."""
from typing import Tuple

from functools import partial
import numpy as np

import gmeep as gm
from gmeep.config import disable_print, enable_print
from gmeep.materials import get_index
from gmeep.types import Mode


def find_modes_dispersion(
    wavelength: float = 1.55,
    wavelength_step: float = 0.01,
    core: str = "Si",
    clad: str = "SiO2",
    **kwargs,
) -> Tuple[float, float]:
    """Returns mode effective index and group index.

    Computes dispersion with finite difference.


    Args:
        wavelength_step: in um
        mode_solver: you can pass an mpb.ModeSolver
        mode_number: to compute
        wg_thickness: wg height (um)
        ncore: core material refractive index
        nclad: clad material refractive index
        sx: supercell width (um)
        sy: supercell height (um)
        res: (pixels/um)
        wavelength: wavelength
        num_bands: mode order
        plot: if True plots mode
        logscale: plots in logscale
        plotH: plot magnetic field
        dirpath: path to save the modes
        polarization: prefix when saving the modes
        paririty: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM. Reduces spurious modes.

    Returns:
        neff
        ng
    """
    w0 = wavelength - wavelength_step
    wc = wavelength
    w1 = wavelength + wavelength_step

    ncore = partial(get_index, name=core)
    nclad = partial(get_index, name=clad)

    disable_print()
    m0 = gm.find_modes(wavelength=w0, ncore=ncore(w0), nclad=nclad(w0), **kwargs)
    mc = gm.find_modes(wavelength=wc, ncore=ncore(wc), nclad=nclad(wc), **kwargs)
    m1 = gm.find_modes(wavelength=w1, ncore=ncore(w1), nclad=nclad(w1), **kwargs)
    enable_print()

    n0 = m0.neff
    nc = mc.neff
    n1 = m1.neff

    # ng = ncenter - wavelength *dn/ step
    ng = nc - wavelength * (n1 - n0) / (2 * wavelength_step)
    neff = (n0 + nc + n1) / 3
    return Mode(ng=ng, neff=neff, solver=mc)


def test_ng():
    m = find_modes_dispersion(wg_width=0.45, wg_thickness=0.22)
    # print(r["ng"])
    assert np.isclose(m.ng, 4.243284836138521)


if __name__ == "__main__":
    test_ng()
    # print(get_index(name="Si"))
    # ngs = []
    # for wavelength_step in [0.001, 0.01]:
    #     neff, ng = find_modes_dispersion(
    #         wg_width=0.45, wg_thickness=0.22, wavelength_step=wavelength_step
    #     )
    #     ngs.append(ng)
    #     print(wavelength_step, ng)
