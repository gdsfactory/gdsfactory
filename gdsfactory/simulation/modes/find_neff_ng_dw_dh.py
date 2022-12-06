"""Compute group and effective index for different waveguide widths and heights.

Reproduce Yufei thesis results with MPB.

https://www.photonics.intec.ugent.be/contact/people.asp?ID=332

"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydantic
from scipy.interpolate import interp2d

from gdsfactory.config import PATH
from gdsfactory.simulation.modes.find_mode_dispersion import find_mode_dispersion

PATH.modes = pathlib.Path.cwd() / "data"

nm = 1e-3
width0 = 465 * nm
thickness0 = 215 * nm


@pydantic.validate_arguments
def find_neff_ng_dw_dh(
    width: float = width0,
    thickness: float = thickness0,
    delta_width: float = 30 * nm,
    delta_thickness: float = 20 * nm,
    wavelength: float = 1.55,
    steps: int = 11,
    mode_number: int = 1,
    core: str = "Si",
    clad: str = "SiO2",
    **kwargs,
) -> pd.DataFrame:
    """Computes group and effective index for different widths and heights.

    Args:
        width: nominal waveguide width in um.
        thickness: nominal waveguide thickness in um.
        delta_width: delta width max in um.
        delta_thickness: delta thickness max in um.
        wavelength: center wavelength (um).
        steps: number of steps to sweep in width and thickness.
        mode_number: mode index to compute (1: fundamental mode).
        core: core material name.
        clad: clad material name.

    Keyword Args:
        wg_thickness: wg height (um).
        sx: supercell width (um).
        sy: supercell height (um).
        resolution: (pixels/um).
        wavelength: wavelength in um.
        num_bands: mode order.
        plot: if True plots mode.
        logscale: plots in logscale.
        plotH: plot magnetic field.
        cache: path to save the modes.
        polarization: prefix when saving the modes.
        parity: symmetries mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.

    """
    dw = np.linspace(-delta_width, delta_width, steps)
    dh = np.linspace(-delta_thickness, delta_thickness, steps)

    neffs = []
    ngs = []
    dhs = []
    dws = []

    for dwi in dw:
        for dhi in dh:
            m = find_mode_dispersion(
                core=core,
                clad=clad,
                wg_width=width + dwi,
                wg_thickness=thickness + dhi,
                wavelength=wavelength,
                mode_number=mode_number,
                **kwargs,
            )
            neffs.append(m.neff)
            ngs.append(m.ng)
            dws.append(dwi)
            dhs.append(dhi)

    return pd.DataFrame(dict(dw=dws, dh=dhs, neff=neffs, ng=ngs))


def plot_neff_ng_dw_dh(
    width: float = width0,
    thickness: float = thickness0,
    wavelength: float = 1.55,
    mode_number: int = 1,
    **kwargs,
) -> None:
    """Plot neff and group index versus width (dw) and height (dh) variations.

    Args:
        width: waveguide width in um.
        thickness: waveguide thickness in um.
        wavelength: in um.
        mode_number: 1 is the fundamental first order mode.

    """
    filepath = pathlib.Path(PATH.modes / "mpb_dw_dh_dispersion.csv")
    m = find_mode_dispersion(
        wg_width=width, wg_thickness=thickness, wavelength=wavelength
    )
    neff0 = m.neff
    ng0 = m.ng

    if filepath.exists():
        df = pd.read_csv(filepath)
    else:
        df = find_neff_ng_dw_dh(wavelength=wavelength, **kwargs)
        cache = filepath.parent
        cache.mkdir(exist_ok=True, parents=True)
        df.to_csv(filepath)

    dws = df.dw.values
    dhs = df.dh.values
    ngs = df.ng.values
    neffs = df.neff.values

    # neff interpolation
    f_w = interp2d(neffs, ngs, np.array(dws), kind="cubic")
    f_h = interp2d(neffs, ngs, np.array(dhs), kind="cubic")

    ws = width + np.array(dws)
    hs = thickness + np.array(dhs)

    plt.plot(ws * 1e3, hs * 1e3, "ko")
    extracted_dw = []
    extracted_dh = []

    for neff, ng in zip(neffs, ngs):
        temp_w = f_w(neff, ng) + width
        temp_h = f_h(neff, ng) + thickness
        extracted_dw.append(temp_w * 1e3)
        extracted_dh.append(temp_h * 1e3)

    plt.plot(extracted_dw, extracted_dh, "rx")
    plt.xlabel("width (nm)")
    plt.ylabel("height (nm)")

    plt.figure()
    plt.plot(neffs, ngs, "ro")
    plt.plot(neff0, ng0, "bx")
    plt.xlabel("neff")
    plt.ylabel("ng")
    plt.show()


if __name__ == "__main__":
    plot_neff_ng_dw_dh()
