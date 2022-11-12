from simphony.models import Subcircuit
from simphony.simulators import SweepSimulator


def get_transmission(
    subcircuit: Subcircuit,
    pin_in: str = "o1",
    pin_out: str = "o2",
    start: float = 1500e-9,
    stop: float = 1600e-9,
    num: int = 2000,
):
    """Return transmission for a circuit.

    Args:
        circuit: for transmission.
        pin_in: input pin.
        pin_out: output pin.
        start: start wavelength (m).
        stop: stop wavelength (m).
        num: number of points.

    """
    simulation = SweepSimulator(start, stop, num)
    simulation.multiconnect(
        subcircuit.circuit.pins[pin_in], subcircuit.circuit.pins[pin_out]
    )
    w, s = simulation.simulate()
    w *= 1e9

    return dict(wavelengths=w, s=s)
