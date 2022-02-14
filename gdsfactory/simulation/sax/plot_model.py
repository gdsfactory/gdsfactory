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
    """Plot Model Sparameters

    Args:
        model: function that returns SDict
        port1: input port name
        ports2: list of ports
        wl_min: wavelength min (um)
        wl_max: wavelength max (um)
        wl_steps: number of wavelength steps
        logscale: plots in dB.
        phase: plots phase instead of magnitude

    .. plot::
        :include-source:

        import gdsfactory.simulation.sax as gs
        import matplotlib.pyplot as plt

        gs.plot_model(gs.models.straight, phase=True, port1="o1")
        plt.show()
    """

    wavelengths = np.linspace(wl_min, wl_max, wl_steps)
    sdict = sdict(wl=wavelengths)

    ports = {ports[0] for ports in sdict.keys()}
    ports2 = ports2 or ports

    if port1 not in ports:
        raise ValueError(f"port1 {port1!r} not in {list(ports)}")

    for port in ports2:
        if port not in ports:
            raise ValueError(f"port2 {port!r} not in {list(ports)}")

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
            ax.plot(wavelengths * 1e9, y, label=port2)
    ax.set_title(port1)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(ylabel)
    plt.legend()
    return ax


if __name__ == "__main__":
    import sax

    plot_model(sax.models.straight, phase=True, port1="in2")
    plt.show()
