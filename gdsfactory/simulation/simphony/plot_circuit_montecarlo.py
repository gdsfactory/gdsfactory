from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from simphony.models import Subcircuit
from simphony.simulators import MonteCarloSweepSimulator


def plot_circuit_montecarlo(
    circuit: Subcircuit,
    pin_in: str = "o1",
    pin_out: str = "o2",
    start: float = 1500e-9,
    stop: float = 1600e-9,
    num: int = 2000,
    logscale: bool = True,
    runs: int = 10,
) -> None:
    """Plot MonteCarlo simulations variation.

    Args:
        circuit: for simulating.
        pin_in: input port name.
        pin_out: iterable of pins out to plot.
        start: wavelength (m).
        stop: wavelength (m).
        num: number of sampled points.
        logscale: plot in dB scale.
        runs: number of Monte Carlo iterations to run.

    .. plot::
        :include-source:

        from gdsfactory.simulation.simphony.components.mzi import mzi
        import gdsfactory.simulation.simphony as gs

        c = mzi()
        gs.plot_circuit_montecarlo(c)

    """
    circuit = circuit() if callable(circuit) else circuit
    simulation = MonteCarloSweepSimulator(start=start, stop=stop, num=num)
    simulation.multiconnect(circuit.pins[pin_in], circuit.pins[pin_out])
    result = simulation.simulate(runs=runs)

    for wl, s in result:
        s = 10 * np.log10(abs(s)) if logscale else abs(s)
        plt.plot(wl, s)

    # The data located at the 0 position is the ideal values.
    wl, s = result[0]
    plt.plot(wl, s, "k")
    plt.title("MZI Monte Carlo")
    ylabel = "|S| (dB)" if logscale else "|S|"
    plt.xlabel("wavelength (m)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from gdsfactory.simulation.simphony.components.mzi import mzi

    plot_circuit_montecarlo(mzi)
    plt.show()
