import pathlib

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic

import gdsfactory as gf
from gdsfactory.simulation.modes.find_modes import find_modes as find_modes_function
from gdsfactory.simulation.modes.get_mode_solver_coupler import get_mode_solver_coupler
from gdsfactory.simulation.modes.types import ModeSolverFactory
from gdsfactory.types import Callable, Optional, PathType


def coupling_length(
    neff1: float,
    neff2: float,
    power_ratio: float = 1.0,
    wavelength: float = 1.55,
) -> float:
    """
    Returns the coupling length (um) of the directional coupler
    to achieve power_ratio

    Args:
        wavelength: in um
        neff1: even supermode of the directional coupler.
        neff2: odd supermode of the directional coupler.
        power_ratio: p2/p1, where 1 means 100% power transfer

    """
    dneff = (neff1 - neff2).real
    return wavelength / (np.pi * dneff) * np.arcsin(np.sqrt(power_ratio))


@pydantic.validate_arguments
def find_coupling(
    gap: float = 0.2,
    mode_solver: ModeSolverFactory = get_mode_solver_coupler,
    find_modes: Callable = find_modes_function,
    power_ratio: float = 1.0,
    wavelength: float = 1.55,
    **kwargs
) -> float:
    """
    Returns the coupling length (um) of the directional coupler
    to achieve power_ratio

    Args:
        gap: in um
        mode_solver: function to get the mode solver
        find_modes: function to find the modes
        power_ratio: p2/p1, where 1 means 100% power transfer
        wavelength: in um

    keyword Args:
        nmodes: number of modes
        parity: for symmetries
    """
    modes = find_modes(
        mode_solver=gf.partial(mode_solver, gaps=(gap,)),
        wavelength=wavelength,
        **kwargs
    )
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
    mode_solver: ModeSolverFactory = get_mode_solver_coupler,
    find_modes: Callable = find_modes_function,
    nmodes: int = 4,
    wavelength: float = 1.55,
    parity=mp.NO_PARITY,
    filepath: Optional[PathType] = None,
    overwrite: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Returns coupling vs gap

    Args:
        gap1:
        gap2:
        steps:
        mode_solver:
        find_modes:
        nmodes:
        wavelength:
        parity:
        filepath:
        overwrite:
    """
    if filepath and not overwrite and pathlib.Path(filepath).exists():
        return pd.read_csv(filepath)

    gaps = np.linspace(gap1, gap2, steps)
    ne = []
    no = []
    gap_to_modes = {}

    for gap in gaps:
        modes = find_modes(mode_solver=gf.partial(mode_solver, gaps=(gap,)))
        ne.append(modes[1].neff)
        no.append(modes[2].neff)
        gap_to_modes[gap] = modes

    lc = [
        coupling_length(neff1=neff1, neff2=neff2)
        for gap, neff1, neff2 in zip(gaps, ne, no)
    ]

    df = pd.DataFrame(dict(gap=gaps, ne=ne, no=no, lc=lc))
    if filepath:
        filepath = pathlib.Path(filepath)
        dirpath = filepath.parent
        dirpath.mkdir(exist_ok=True, parents=True)
        df.to_csv(filepath, index=False)
    return df


def plot_coupling_vs_gap(df: pd.DataFrame, **kwargs):
    plt.plot(df.gap, df.lc, ".-")
    plt.ylabel("100% coupling length (um)")
    plt.xlabel("gap (um)")


if __name__ == "__main__":
    df = find_coupling_vs_gap(steps=3, filepath="coupling_vs_gap.csv", overwrite=True)
    plot_coupling_vs_gap(df)
    plt.show()
