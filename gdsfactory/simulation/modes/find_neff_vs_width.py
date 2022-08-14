import pathlib

import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import pandas as pd
import pydantic
from tqdm.auto import tqdm

from gdsfactory.simulation.modes.find_modes import find_modes_waveguide
from gdsfactory.types import Optional, PathType


@pydantic.validate_arguments
def find_neff_vs_width(
    width1: float = 0.2,
    width2: float = 1.0,
    steps: int = 12,
    nmodes: int = 4,
    wavelength: float = 1.55,
    parity=mp.NO_PARITY,
    filepath: Optional[PathType] = None,
    overwrite: bool = False,
    **kwargs
) -> pd.DataFrame:
    """Sweep waveguide width and compute effective index.

    Args:
        width1: starting waveguide width in um.
        width2: end waveguide width in um.
        steps: number of points.
        nmodes: number of modes to compute.
        wavelength: wavelength in um.
        parity: mp.ODD_Y mp.EVEN_X for TE, mp.EVEN_Y for TM.
        filepath: Optional filepath to store the results.
        overwrite: overwrite file even if exists on disk.


    Keyword Args:
        slab_thickness: thickness for the waveguide slab in um.
        ncore: core material refractive index.
        nclad: clad material refractive index.
        sy: simulation region width (um).
        sz: simulation region height (um).
        resolution: resolution (pixels/um).

    """
    if filepath and not overwrite and pathlib.Path(filepath).exists():
        return pd.read_csv(filepath)

    width = np.linspace(width1, width2, steps)
    neff = {mode_number: [] for mode_number in range(1, nmodes + 1)}
    for wg_width in tqdm(width):
        modes = find_modes_waveguide(
            wavelength=wavelength,
            parity=parity,
            nmodes=nmodes,
            wg_width=wg_width,
            **kwargs
        )
        for mode_number in range(1, nmodes + 1):
            mode = modes[mode_number]
            neff[mode_number].append(mode.neff)

    df = pd.DataFrame(neff)
    df["width"] = width
    if filepath:
        filepath = pathlib.Path(filepath)
        cache = filepath.parent
        cache.mkdir(exist_ok=True, parents=True)
        df.to_csv(filepath, index=False)
    return df


def plot_neff_vs_width(df: pd.DataFrame, **kwargs) -> None:
    """Plots effective index versus waveguide width."""
    width = df.width
    for mode_number, neff in df.items():
        if mode_number != "width":
            plt.plot(width, neff, ".-", label=str(mode_number))

    plt.legend(**kwargs)
    plt.xlabel("width (um)")
    plt.ylabel("neff")


if __name__ == "__main__":
    df = find_neff_vs_width(steps=3, filepath="neff_vs_width.csv")
    plot_neff_vs_width(df)
    plt.show()
