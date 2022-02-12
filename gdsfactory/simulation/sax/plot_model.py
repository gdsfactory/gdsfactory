""" Useful functions for plotting. """

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sax.typing_ import Model


def plot_model(
    sdict: Model,
    port1: str = "o1",
    ports2: Tuple[str, ...] = None,
    logscale: bool = True,
    fig=None,
    wl_min: float = 1.5,
    wl_max: float = 1.6,
    wl_steps: int = 2000,
    phase: bool = False,
) -> None:
    """Plot Sparameters for a model

    Args:
        model: simphony model
        port1: input pin name
        pins: list of pins
        wl_min: wavelength min (um)
        wl_max: wavelength max (um)
        wl_steps: number of steps
        logscale:
        fig: figure
        phase: plots phase instead of magnitude

    .. plot::
        :include-source:

        import gdsfactory.simulation.sax as gs
        import matplotlib.pyplot as plt

        gs.plot_model(sax.models.pic.straight, phase=True, port1="in0")
        plt.show()
    """

    wavelengths = np.linspace(wl_min, wl_max, wl_steps)
    sdict = sdict(wl=wavelengths)

    ports = {ports[0] for ports in sdict.keys()}
    ports2 = ports2 or ports

    for port in ports2:
        if port not in ports:
            raise ValueError(f"port {port!r} not in {ports}")

    fig = fig or plt.subplot()
    ax = fig.axes

    for port2 in ports2:
        if (port1, port2) in sdict:
            if phase:
                y = np.angle(sdict[(port1, port2)])
                ylabel = "angle (rad)"
            else:
                y = np.abs(sdict[(port1, port2)])
                y = 20 * np.log10(y) if logscale else y
                ylabel = "|S (dB)|" if logscale else "|S|"
            ax.plot(wavelengths * 1e9, y, label=f"{port1}->{port2}")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(ylabel)
    plt.legend()
    return ax


if __name__ == "__main__":
    import sax

    plot_model(sax.models.straight, phase=True, port1="in0")
    plt.show()
