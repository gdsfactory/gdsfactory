"""[[-0.09051371-0.20581339j  0.00704022+0.1328474j \
    0.03733851+0.4879802j ]."""

from pathlib import PosixPath
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from simphony import Model
from simphony.tools import freq2wl, interpolate, wl2freq

import gdsfactory as gf
from gdsfactory.simulation.lumerical.read import read_sparameters_file


class SimphonyFromFile(Model):
    """Take an s-parameter file path and return a Simphony Model from it."""

    def __init__(
        self,
        numports: Optional[int] = None,
        pins: Optional[Tuple[str, ...]] = None,
        name: str = "model",
    ) -> None:
        """Take an s-parameter file path and return a Simphony Model from it.

        Uses numports if passed, else uses pins.

        Args:
            numports: number of ports.
            pins: list of port names.
            name: optional model name.
        """
        if numports:
            __class__.pin_count = numports
        elif pins:
            __class__.pin_count = len(pins)
        else:
            raise ValueError("Either numports or pin must be defined.")

        super().__init__(name)

        if pins:
            self.rename_pins(*pins)

    def model_from_filepath(
        self,
        filepath: PosixPath,
    ) -> Model:
        """Returns simphony Model from lumerical .DAT sparameters.

        Args:
            filepath: path to Sparameters in Lumerical interconnect format.

        """
        numports = self.pin_count
        pin_names, self.f, self.s = read_sparameters_file(
            filepath=filepath, numports=numports
        )
        self.wavelengths = freq2wl(self.f)

        self.rename_pins(*pin_names)

        return self

    def model_from_csv(
        self,
        filepath: Union[str, PosixPath, pd.DataFrame],
        xkey: str = "wavelengths",
        xunits: float = 1,
    ) -> Model:
        """Returns simphony Model from Sparameters in CSV.

        Args:
            filepath: CSV Sparameters path or pandas DataFrame.
            xkey: key for wavelengths in file.
            xunits: x units in um from the loaded file (um). 1 means 1um.

        """
        df = filepath if isinstance(filepath, pd.DataFrame) else pd.read_csv(filepath)

        keys = list(df.keys())

        if xkey not in df:
            raise ValueError(f"{xkey!r} not in {keys}")

        self.wavelengths = df[xkey].values * 1e-6 * xunits

        numrows = len(self.wavelengths)
        numports = self.pin_count
        self.s = np.zeros((numrows, numports, numports), dtype="complex128")

        for i in range(numports):
            for j in range(numports):
                self.s[:, i, j] = df[f"s{i+1}{j+1}m"] * np.exp(1j * df[f"s{i+1}{j+1}a"])

        self.f = wl2freq(self.wavelengths)
        self.freq_range = (self.f[0], self.f[-1])

        return self

    def s_parameters(self, freqs: "np.array") -> "np.ndarray":
        return interpolate(freqs, self.f, self.s)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_model import plot_model

    filepath = gf.CONFIG["sparameters"] / "mmi1x2" / "mmi1x2_si220n.dat"
    numports = 3
    c = SimphonyFromFile(numports=numports).model_from_filepath(filepath=filepath)

    filepath_csv = gf.CONFIG["sparameters"] / "mmi1x2" / "mmi1x2_si220n.csv"
    # filepath_csv = "/home/jmatres/ubc/sparameters/ebeam_y_1550_20634f71.csv"
    # df = pd.read_csv(filepath_csv)
    c = SimphonyFromFile(pins=("o1", "o2", "o3")).model_from_csv(filepath_csv)
    plot_model(c, pin_in="o1")
    plt.show()

    # wav = np.linspace(1520, 1570, 1024) * 1e-9
    # f = speed_of_light / wav
    # s = c.s_parameters(freqs=f)
    # wav = c.wavelengths
    # s = c.s
    # plt.plot(wav * 1e9, np.abs(s[:, 1] ** 2))
