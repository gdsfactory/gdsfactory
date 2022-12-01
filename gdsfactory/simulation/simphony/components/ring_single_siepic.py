from __future__ import annotations

import numpy as np
from simphony.libraries import siepic

from gdsfactory.simulation.simphony.components.coupler_ring import coupler_ring


def ring_single(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    length_y=4,
    bend_radius=5,
    coupler=coupler_ring,
    straight=siepic.Waveguide,
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

    cb = coupler
    wt = straight

    cb.rename_pins("W0", "N0", "N1", "E0")
    wt.rename_pins("n1", "n2")

    cb["N0"].connect(wt["n1"])
    cb["N1"].connect(wt["n2"])

    cb["W0"].rename("input")
    cb["E0"].rename("output")

    return cb.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony import plot_circuit

    c = ring_single(length_y=20)
    plot_circuit(c, pin_in="input", pins_out=("output",))
    plt.show()
