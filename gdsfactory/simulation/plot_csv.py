from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import gdsfactory as gf


def plot_sparameters(
    df: DataFrame,
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
) -> None:
    """Plots Sparameters from a pandas DataFrame.

    Args:
        df: Sparameters pandas DataFrame.
        logscale: plots 20*log10(S).
        keys: list of keys to plot, plots all by default.

    .. plot::
        :include-source:

        import gdsfactory as gf
        import gdsfactory.simulation as sim

        df = sim.get_sparameters_data_lumerical(component=gf.components.mmi1x2)
        sim.plot.plot_sparameters(df, logscale=True)

    """
    w = df["wavelengths"] * 1e3
    keys = keys or [
        key for key in df.keys() if key.lower().startswith("s") and key.endswith("m")
    ]

    for key in keys:
        if key not in df:
            raise ValueError(f"{key} not in {df.keys()}")
        y = df[key]
        y = 20 * np.log10(y) if logscale else y
        plt.plot(w, y, label=key[:-1])
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("|S| (dB)") if logscale else plt.ylabel("|S|")
    plt.show()


def plot_imbalance2x2(df: DataFrame, port1: str = "s13m", port2: str = "s14m") -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        df: pandas DataFrame.
        port1: name.
        port2: name.

    """
    y1 = df[port1].values
    y2 = df[port2].values
    imbalance = y1 / y2
    x = df["wavelengths"] * 1e3
    plt.plot(x, 100 * abs(imbalance))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("imbalance (%)")
    plt.grid()


def plot_loss2x2(df: DataFrame, port1: str = "s13m", port2: str = "s14m") -> None:
    """Plots imbalance in % for 2x2 coupler.

    Args:
        df: pandas DataFrame.
        port1: name.
        port2: name.

    """
    y1 = df[port1].values
    y2 = df[port2].values
    x = df["wavelengths"] * 1e3
    plt.plot(x, abs(10 * np.log10(y1**2 + y2**2)))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("excess loss (dB)")


plot_loss1x2 = gf.partial(plot_loss2x2, port1="s13m", port2="o1@0,o2@0")
plot_imbalance1x2 = gf.partial(plot_imbalance2x2, port1="s13m", port2="s12m")


if __name__ == "__main__":
    import gdsfactory.simulation as sim

    df = sim.get_sparameters_data_lumerical(component=gf.components.mmi1x2)
    plot_sparameters(df, logscale=True)
    plt.show()
