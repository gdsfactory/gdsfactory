import matplotlib.pyplot as plt
import numpy as np

from gdsfactory.simulation.mpb.find_modes import find_modes


def find_modes_sweep_width(wg_widths, **kwargs):
    """Returns effective index and group index for a mode.

    Args:
        wg_widths: list of widths (um)
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
        prefix: prefix when saving the modes

    Returns:
        neff
        ng
    """
    n = [find_modes(wg_width=wg_width, **kwargs) for wg_width in wg_widths]
    neffs = [ni["neff"] for ni in n]
    ngs = [ni["ng"] for ni in n]
    return neffs, ngs


if __name__ == "__main__":
    wg_widths = np.arange(200, 2000, 200) * 1e-3
    neffs, ngs = find_modes_sweep_width(wg_widths=wg_widths, mode_number=1)

    plt.figure()
    plt.plot(wg_widths, neffs, ".-")
    plt.xlabel("wg_width")
    plt.ylabel("neff")

    plt.figure()
    plt.plot(wg_widths, ngs, ".-")
    plt.xlabel("wg_width")
    plt.ylabel("ng")
    plt.show()
