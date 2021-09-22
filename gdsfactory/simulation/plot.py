from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.simulation.write_sparameters_lumerical import (
    write_sparameters_lumerical,
)


def plot_sparameters(
    component_or_df: Union[Component, DataFrame],
    logscale: bool = True,
    keys: Optional[Tuple[str, ...]] = None,
    dirpath: Path = gf.CONFIG["sparameters"],
    write_sparameters_function: Callable = write_sparameters_lumerical,
    **sim_settings,
):
    """Plots Sparameters.

    Args:
        component_or_df: Component or Sparameters in pandas DataFrame
        logscale: plots 20*log10(results)
        keys: list of keys to plot
        dirpath: where to store the simulations
        **sim_settings: simulation kwargs

    """

    r = component_or_df
    if isinstance(r, Component):
        r = write_sparameters_function(component=r, dirpath=dirpath, **sim_settings)
    w = r["wavelength_nm"]

    if keys:
        keys = [key for key in keys if key in r.keys()]
    else:
        keys = [key for key in r.keys() if key.startswith("S") and key.endswith("m")]

    for key in keys:
        y = 20 * np.log10(r[key]) if logscale else r[key]
        plt.plot(w, y, label=key[:-1])
    plt.legend()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Transmission (dB)") if logscale else plt.ylabel("Transmission")


if __name__ == "__main__":

    remove_layers = []
    layer_to_thickness = {(1, 0): 220e-3}

    # r = write(component=gf.components.straight(), layer_to_thickness=layer_to_thickness)
    # r = write(component=gf.components.mmi2x2(), layer_to_thickness=layer_to_thickness)
    # r = write(component=gf.components.mmi1x2(), layer_to_thickness=layer_to_thickness)
    # r = write(component=gf.components.coupler(), layer_to_thickness=layer_to_thickness)
    # r = write(component=gf.components.bend_circular(), layer_to_thickness=layer_to_thickness)
    # plot_sparameters(r, logscale=True)
    # plot_sparameters(gf.components.coupler())
    plot_sparameters(gf.components.mmi1x2(), logscale=False)
    plt.show()
