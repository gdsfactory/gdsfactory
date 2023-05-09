from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic
from tqdm.auto import tqdm

from gdsfactory.simulation.modes.find_modes import find_modes_coupler
from gdsfactory.typings import Optional, PathType


def coupling_length(
    neff1: float,
    neff2: float,
    power_ratio: float = 1.0,
    wavelength: float = 1.55,
) -> float:
    """Returns the coupling length (um) of the directional coupler to achieve power_ratio.

    Args:
        wavelength: in um.
        neff1: even supermode of the directional coupler.
        neff2: odd supermode of the directional coupler.
        power_ratio: p2/p1, where 1 means 100% power transfer.

    """
    dneff = (neff1 - neff2).real
    return wavelength / (np.pi * dneff) * np.arcsin(np.sqrt(power_ratio))


@pydantic.validate_arguments
def find_coupling(
    gap: float = 0.2, power_ratio: float = 1.0, wavelength: float = 1.55, **kwargs
) -> float:
    """Returns the coupling length (um) of the directional coupler to achieve power_ratio, where 1 means 100% power transfer.

    Args:
        gap: in um
        power_ratio: p2/p1, where 1 means 100% power transfer
        wavelength: in um

    keyword Args:
        nmodes: number of modes
        parity: for symmetries

    """
    modes = find_modes_coupler(gaps=(gap,), wavelength=wavelength, **kwargs)
    ne = modes[1].neff
    no = modes[2].neff

    return coupling_length(
        neff1=ne, neff2=no, power_ratio=power_ratio, wavelength=wavelength
    )


@pydantic.validate_arguments
def find_coupling_vs_gap(
    gap1: float = 0.2,
    gap2: float = 0.4,
    steps: int = 12,
    nmodes: int = 4,
    wavelength: float = 1.55,
    parity=mp.NO_PARITY,
    filepath: Optional[PathType] = None,
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Returns coupling vs gap pandas DataFrame.

    Args:
        gap1: starting gap in um.
        gap2: end gap in um.
        steps: number of steps.
        nmodes: number of modes.
        wavelength: wavelength (um).
        parity: for symmetries.
        filepath: optional filepath to cache results on disk.
        overwrite: overwrites results even if found on disk.

    Keyword Args:
        core_width: core_width (um) for the symmetric case.
        gap: for the case of only two waveguides.
        core_widths: list or tuple of waveguide widths.
        gaps: list or tuple of waveguide gaps.
        core_thickness: wg height (um)
        slab_thickness: thickness for the waveguide slab
        core_material: core material refractive index
        clad_material: clad material refractive index
        nslab: Optional slab material refractive index. Defaults to core_material.
        ymargin: margin in y.
        sz: simulation region thickness (um).
        resolution: resolution (pixels/um).
        nmodes: number of modes.
        sidewall_angles: waveguide sidewall angle (radians),
            tapers from core_width at top of slab, upwards, to top of waveguide.

    """
    if filepath and not overwrite and pathlib.Path(filepath).exists():
        return pd.read_csv(filepath)

    gaps = np.linspace(gap1, gap2, steps)
    ne = []
    no = []
    dn = []
    lc = []

    for gap in tqdm(gaps):
        modes = find_modes_coupler(gaps=(gap,), **kwargs)
        n1 = modes[1].neff
        n2 = modes[2].neff
        coupling = coupling_length(n1, n2)

        ne.append(n1)
        no.append(n2)
        dn.append(n1 - n2)
        lc.append(coupling)

    df = pd.DataFrame(dict(gap=gaps, ne=ne, no=no, lc=lc, dn=dn))
    if filepath:
        filepath = pathlib.Path(filepath)
        cache = filepath.parent
        cache.mkdir(exist_ok=True, parents=True)
        df.to_csv(filepath, index=False)
    return df


def plot_coupling_vs_gap(df: pd.DataFrame, **kwargs) -> None:
    plt.plot(df.gap, df.lc, ".-")
    plt.ylabel("100% coupling length (um)")
    plt.xlabel("gap (um)")
    plt.show()


if __name__ == "__main__":
    df = find_coupling_vs_gap(steps=3, filepath="coupling_vs_gap.csv", overwrite=True)
    plot_coupling_vs_gap(df)
    plt.show()
