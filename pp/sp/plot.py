from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

import pp
from pp.component import Component
from pp.sp.write import write


def plot(
    component_or_df: Union[Component, DataFrame],
    logscale: bool = True,
    keys: Optional[Iterable[str]] = None,
    dirpath: Path = pp.CONFIG["sp"],
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
        r = write(component=r, dirpath=dirpath, **sim_settings)
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
    layer2nm = {(1, 0): 220}

    # r = write(component=pp.c.waveguide(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi2x2(), layer2nm=layer2nm)
    # r = write(component=pp.c.mmi1x2(), layer2nm=layer2nm)
    # r = write(component=pp.c.coupler(), layer2nm=layer2nm)
    # r = write(component=pp.c.bend_circular(), layer2nm=layer2nm)
    # plot(r, logscale=True)
    # plot(pp.c.coupler())
    plot(pp.c.mmi1x2(), logscale=False)
    plt.show()
