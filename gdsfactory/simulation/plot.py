from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import gdsfactory as gf
from gdsfactory.simulation.write_sparameters_lumerical import (
    write_sparameters_lumerical,
)
from gdsfactory.types import Component, ComponentOrFactory


def plot_sparameters(
    component_or_df: Union[ComponentOrFactory, DataFrame],
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
    dirpath: Path = gf.CONFIG["sparameters"],
    write_sparameters_function: Callable = write_sparameters_lumerical,
    **sim_settings,
):
    """Plots Sparameters.

    Args:
        component_or_df: Component or Sparameters pandas DataFrame
        logscale: plots 20*log10(results)
        keys: list of keys to plot, plots all by default
        dirpath: where to store/read the simulations
        write_sparameters_function: custom function to write sparameters

    Keyword Args:
        sim_settings: simulation settings for the write_sparameters_function

    """

    df = component_or_df() if callable(component_or_df) else component_or_df
    if isinstance(df, Component):
        df = write_sparameters_function(component=df, dirpath=dirpath, **sim_settings)
    w = df["wavelengths"] * 1e3

    keys = keys or [
        key for key in df.keys() if key.startswith("s") and key.endswith("m")
    ]

    for key in keys:
        if key in df:
            y = df[key]
            y = 20 * np.log10(y) if logscale else y
            plt.plot(w, y, label=key[:-1])
        else:
            raise ValueError(f"{key} not in {df.keys()}")
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("|S| (dB)") if logscale else plt.ylabel("|S|")


def plot_imbalance2x2(df: DataFrame, port1: str = "S13m", port2: str = "S14m") -> None:
    """Plots imbalance in % for 2x2 coupler"""
    y1 = df[port1].values
    y2 = df[port2].values
    imbalance = y1 / y2
    plt.plot(df.wavelength_nm, 100 * abs(imbalance))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("imbalance (%)")
    plt.grid()


def plot_loss2x2(df: DataFrame, port1: str = "S13m", port2: str = "S14m") -> None:
    """Plots imbalance in % for 2x2 coupler"""
    y1 = df[port1].values
    y2 = df[port2].values
    plt.plot(df.wavelength_nm, abs(10 * np.log10(y1 ** 2 + y2 ** 2)))
    plt.xlabel("wavelength")
    plt.ylabel("excess loss (dB)")


plot_loss1x2 = gf.partial(plot_loss2x2, port1="S13m", port2="S12m")
plot_imbalance1x2 = gf.partial(plot_imbalance2x2, port1="S13m", port2="S12m")


if __name__ == "__main__":
    # plot_sparameters(df, logscale=True)
    # plot_sparameters(gf.components.coupler())
    plot_sparameters(gf.components.mmi1x2(), logscale=False)
    plt.show()
