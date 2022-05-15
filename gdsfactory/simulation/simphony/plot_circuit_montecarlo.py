import matplotlib.pyplot as plt
import numpy as np
from simphony.netlist import Subcircuit
from simphony.simulation import MonteCarloSweepSimulation
from simphony.tools import freq2wl


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
    """Plot MonterCarlo simulations variation.

    Args:
        circuit:
        pin_in: input port name
        pin_out: iterable of pins out to plot
        start: wavelength (m)
        stop: wavelength (m)
        num: number of sampled points
        logscale: plot in dB scale
        runs: number of runs

    """
    circuit = circuit() if callable(circuit) else circuit
    simulation = MonteCarloSweepSimulation(circuit, start=start, stop=stop, num=num)
    result = simulation.simulate(runs=runs)

    for i in range(1, runs + 1):
        f, s = result.data(pin_in, pin_out, i)
        wl = freq2wl(f)
        s = 10 * np.log10(abs(s)) if logscale else abs(s)
        plt.plot(wl, s)

    # The data located at the 0 position is the ideal values.
    f, s = result.data(pin_in, pin_out, 0)
    wl = freq2wl(f)
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
