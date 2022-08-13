from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from simphony.models import Subcircuit
from simphony.simulators import SweepSimulator


def plot_circuit(
    circuit: Subcircuit,
    pin_in: str = "o1",
    pins_out: Tuple[str, ...] = ("o2",),
    start: float = 1500e-9,
    stop: float = 1600e-9,
    num: int = 2000,
    logscale: bool = True,
    fig: Optional[plt.Figure] = None,
    phase: bool = False,
) -> None:
    """Plot Sparameter circuit transmission over wavelength.

    Args:
        circuit: to plot.
        pin_in: input port name.
        pins_out: iterable of pins out to plot.
        start: wavelength (m).
        stop: wavelength (m).
        num: number of sampled points.
        logscale: plot in dB scale.
        fig: matplotlib figure.
        phase: plots phase instead of module.

    .. plot::
        :include-source:

        from gdsfactory.simulation.simphony.components.mzi import mzi
        import gdsfactory.simulation.simphony as gs

        c = mzi()
        gs.plot_circuit(c)

    """
    if not isinstance(pins_out, (set, list, tuple)):
        raise ValueError("pins out is not iterable")

    circuit = circuit() if callable(circuit) else circuit

    fig = fig or plt.subplot()
    ax = fig.axes

    simulation = SweepSimulator(start, stop, num)

    for p in pins_out:
        simulation.multiconnect(circuit.pins[pin_in], circuit.pins[p])
        wl, s = simulation.simulate()

        wl *= 1e9

        if phase:
            y = np.angle(s)
            ylabel = "angle (rad)"
        else:
            y = np.abs(s)
            y = 10 * np.log10(y) if logscale else y
            ylabel = "|S|" if logscale else "|S (dB)|"

        ax.plot(wl, y, label=pins_out[0])
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(ylabel)
    if hasattr(circuit, "name"):
        ax.set_title(circuit.name)
    ax.legend()
    plt.show()
    return ax


def demo_single_port() -> None:
    import gdsfactory.simulation.simphony.components as gc

    c = gc.mzi()
    plot_circuit(c, logscale=False)
    plt.show()


if __name__ == "__main__":
    from gdsfactory.simulation.simphony.components.mzi import mzi

    # import gdsfactory.simulation.simphony.components as gc
    # c = gc.ring_double()
    # plot_circuit(c, pins_out=("cdrop", "drop", "output", "input"))

    c = mzi()
    plot_circuit(c)
    plt.show()
