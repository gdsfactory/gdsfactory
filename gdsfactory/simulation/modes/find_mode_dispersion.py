"""Sweep neff over wavelength and returns group index."""

from functools import partial

from gdsfactory.simulation.gmeep.get_material import get_index
from gdsfactory.simulation.modes.find_modes import find_modes
from gdsfactory.simulation.modes.types import Mode


def find_mode_dispersion(
    wavelength: float = 1.55,
    wavelength_step: float = 0.01,
    core: str = "Si",
    clad: str = "SiO2",
    mode_number: int = 1,
    **kwargs,
) -> Mode:
    """Returns Mode with correct dispersion (ng)

    group index comes from a finite difference approximation at 3 wavelengths

    Args:
        wavelength: center wavelength
        wavelength_step: in um
        core: core material name
        clad: clad material name
        mode_number: to compute
        kwargs:
            wg_thickness: wg height (um)
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
            paririty: symmetries mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM

    """
    w0 = wavelength - wavelength_step
    wc = wavelength
    w1 = wavelength + wavelength_step

    ncore = partial(get_index, name=core)
    nclad = partial(get_index, name=clad)

    m0 = find_modes(wavelength=w0, ncore=ncore(w0), nclad=nclad(w0), **kwargs)
    mc = find_modes(wavelength=wc, ncore=ncore(wc), nclad=nclad(wc), **kwargs)
    m1 = find_modes(wavelength=w1, ncore=ncore(w1), nclad=nclad(w1), **kwargs)

    n0 = m0[mode_number].neff
    nc = mc[mode_number].neff
    n1 = m1[mode_number].neff

    # ng = ncenter - wavelength *dn/ step
    ng = nc - wavelength * (n1 - n0) / (2 * wavelength_step)
    neff = (n0 + nc + n1) / 3
    return Mode(mode_number=mode_number, ng=ng, neff=neff, wavelength=wavelength)


if __name__ == "__main__":
    m = find_mode_dispersion(wg_width=0.45, wg_thickness=0.22)
    print(m.ng)
    # test_ng()
    # print(get_index(name="Si"))
    # ngs = []
    # for wavelength_step in [0.001, 0.01]:
    #     neff, ng = find_modes_dispersion(
    #         wg_width=0.45, wg_thickness=0.22, wavelength_step=wavelength_step
    #     )
    #     ngs.append(ng)
    #     print(wavelength_step, ng)
