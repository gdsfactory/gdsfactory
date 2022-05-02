import numpy as np
from simphony.library import siepic
from simphony.netlist import Subcircuit

from gdsfactory.simulation.simphony.components.coupler_ring import coupler_ring


def ring_single(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    length_y=4,
    bend_radius=5,
    coupler=coupler_ring,
    straight=siepic.ebeam_wg_integral_1550,
):
    r"""Single bus ring made of a ring coupler (cb: bottom).

    .. code::

             -----wt-----
           N0            N1
           |             |
            \           /
             \         /
           ---=========---
        W0    length_x    E0


    """
    length = np.pi * bend_radius + length_x + 2 * length_y
    straight = straight(length=length * 1e-6) if callable(straight) else straight
    coupler = (
        coupler(length_x=length_x, bend_radius=bend_radius, gap=gap, wg_width=wg_width)
        if callable(coupler)
        else coupler
    )

    # Create the circuit, add all individual instances
    circuit = Subcircuit("ring_double")
    circuit.add([(coupler, "cb"), (straight, "wt")])

    circuit.connect_many([("cb", "N0", "wt", "n1"), ("wt", "n2", "cb", "N1")])
    circuit.elements["cb"].pins["W0"] = "input"
    circuit.elements["cb"].pins["E0"] = "output"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulationsimphony import plot_circuit

    c = ring_single(length_y=20)
    plot_circuit(c)
    plt.show()
