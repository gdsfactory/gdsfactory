"""
 [[-0.09051371-0.20581339j  0.00704022+0.1328474j
    0.03733851+0.4879802j ]

"""

from pathlib import PosixPath
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.constants import speed_of_light
from simphony import Model
from simphony.pins import Pin, PinList
from simphony.tools import freq2wl, interpolate, wl2freq

import gdsfactory as gf
from gdsfactory.simulation.lumerical.read import read_sparameters_file


def model_from_filepath(
    filepath: PosixPath, numports: int, name: str = "model"
) -> Model:
    """Returns simphony Model from lumerical .DAT sparameters.

    Args:
        filepath: path to Sparameters in Lumerical interconnect format.
        numports: number of ports.
        name: model name.

    """
    pins, f, s = read_sparameters_file(filepath=filepath, numports=numports)
    wavelengths = freq2wl(f)
    return model_from_sparameters(
        wavelengths=wavelengths, sparameters=s, pins=pins, name=name
    )


def model_from_sparameters(
    wavelengths, sparameters, pins: Tuple[str, ...] = ("E0", "W0"), name: str = "model"
) -> Model:
    """Returns simphony Model from wavelengths and Sparameters.

    Args:
        wavelengths: 1D wavelength array.
        sparameters: numpy nxn array.
        pins: list of port names.
        name: optional model name.
    """
    f = wl2freq(wavelengths)
    s = sparameters

    def interpolate_sp(freq):
        return interpolate(freq, f, s)

    Model.pin_count = len(pins)
    m = Model()
    m.pins = PinList([Pin(m, pname) for i, pname in enumerate(pins)])
    m.s_params = (f, s)
    m.s_parameters = interpolate_sp
    m.freq_range = (min(f), max(f))
    m.wavelength_range = (min(wavelengths), max(wavelengths))
    m.wavelengths = speed_of_light / np.array(f)
    m.s = s
    m.name = name
    m.__name__ = name
    return m


def model_from_csv(
    filepath: Union[str, PosixPath, pd.DataFrame],
    pins: Tuple[str, ...],
    name: str = "model",
    xkey: str = "wavelengths",
    xunits: float = 1,
) -> Model:
    """Returns simphony Model from Sparameters in CSV.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        sparameters: numpy nxn array.
        pins: list of port names.
        name: optional model name.
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
    """

    df = filepath if isinstance(filepath, pd.DataFrame) else pd.read_csv(filepath)

    keys = list(df.keys())

    if xkey not in df:
        raise ValueError(f"{xkey!r} not in {keys}")

    wavelengths = df[xkey].values * 1e-6 * xunits

    numrows = len(wavelengths)
    numports = len(pins)
    S = np.zeros((numrows, numports, numports), dtype="complex128")

    for i in range(len(pins)):
        for j in range(len(pins)):
            S[:, i, j] = df[f"s{i+1}{j+1}m"] * np.exp(1j * df[f"s{i+1}{j+1}a"])

    return model_from_sparameters(
        wavelengths=wavelengths, sparameters=S, pins=pins, name=name
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_model import plot_model

    filepath = gf.CONFIG["sparameters"] / "mmi1x2" / "mmi1x2_si220n.dat"
    numports = 3
    c = model_from_filepath(filepath=filepath, numports=numports)

    filepath_csv = gf.CONFIG["sparameters"] / "mmi1x2" / "mmi1x2_si220n.csv"
    filepath_csv = "/home/jmatres/ubc/sparameters/ebeam_y_1550_20634f71.csv"
    df = pd.read_csv(filepath_csv)
    c = model_from_csv(df, pins=("o1", "o2", "o3"))
    plot_model(c, pin_in="o1")
    plt.show()

    # wav = np.linspace(1520, 1570, 1024) * 1e-9
    # f = speed_of_light / wav
    # s = c.s_parameters(freq=f)
    # wav = c.wavelengths
    # s = c.s
    # plt.plot(wav * 1e9, np.abs(s[:, 1] ** 2))
