from pathlib import PosixPath
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np

import pp
from pp.sp.write import write


def plot(
    component_or_results_dict,
    logscale: bool = True,
    keys: Optional[Iterable[str]] = None,
    dirpath: PosixPath = pp.CONFIG["sp"],
    **sim_settings,
):
    """Plots Sparameters.

    Args:
        component_or_results_dict:
        logscale: plots 20*log10(results)
        keys: list of keys to plot
        dirpath: where to store the simulations
        **sim_settings: simulation kwargs

    """

    r = component_or_results_dict
    if isinstance(r, pp.Component):
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
