from simphony.netlist import Subcircuit
from simphony.simulation import SweepSimulation
from simphony.tools import freq2wl


def get_transmission(
    circuit: Subcircuit,
    pin_in: str = "o1",
    pin_out: str = "o2",
    start: float = 1500e-9,
    stop: float = 1600e-9,
    num: int = 2000,
):
    """Return transmission for a circuit.

    Args:
        circuit:
        pin_in: input pin
        pin_out: output pin
        start: start wavelength (m)
        stop: stop wavelength (m)
        num: number of points

    """
    simulation = SweepSimulation(circuit, start, stop, num)
    result = simulation.simulate()

    f, s = result.data(pin_in, pin_out)
    w = freq2wl(f) * 1e9
    return dict(wavelengths=w, s=s)
