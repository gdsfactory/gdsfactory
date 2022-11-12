"""Useful plot functions."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydantic import validate_arguments
from sax.typing_ import Model


@validate_arguments
def plot_model(
    model: Model,
    port1: str = "o1",
    ports2: Tuple[str, ...] = None,
    logscale: bool = True,
    fig=None,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    wavelength_points: int = 2000,
    phase: bool = False,
) -> None:
    """Plot Model Sparameters Magnitude.

    Args:
        model: function that returns SDict as function of wavelength.
        port1: input port name.
        ports2: list of ports.
        logscale: plots in dB logarithmic scale.
        wavelength_start: wavelength min (um).
        wavelength_stop: wavelength max (um).
        wavelength_points: number of wavelength steps.
        phase: plot phase instead of magnitude.

    .. plot::
        :include-source:

        import gdsfactory.simulation.sax as gs

        gs.plot_model(gs.models.straight, phase=True, port1="o1")

    """
    wavelengths = np.linspace(wavelength_start, wavelength_stop, wavelength_points)
    sdict = model(wl=wavelengths)

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
