from simphony.netlist import Subcircuit

from gdsfactory.simulation.simphony.components.coupler_ring import coupler_ring
from gdsfactory.simulation.simphony.components.straight import (
    straight as straight_function,
)
from gdsfactory.simulation.simphony.plot_circuit import plot_circuit
from gdsfactory.simulation.simphony.types import ModelFactory


def ring_double(
    wg_width: float = 0.5,
    gap: float = 0.2,
    length_x: float = 4,
    radius: float = 5,
    length_y: float = 2,
    coupler: ModelFactory = coupler_ring,
    straight: ModelFactory = straight_function,
) -> Subcircuit:
    r"""Return double bus ring Model made of two couplers (ct: top, cb: bottom).

    connected with two vertical straights (yl: left, wr: right)

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x


           ---=========---
        o2    length_x    o3
             /         \
            /           \
           |             |
           o3           o2 ___
                            |
          wl            wr  | length_y
                           _|_
           o2            o3
           |             |
            \           /
             \         /
           ---=========---
        o1    length_x    o4



    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.ring_double(width=0.5, gap=0.2, length_x=4, radius=5, length_y=2)
      c.plot()


    .. plot::
        :include-source:

        import gdsfactory.simulation simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.ring_double()
        gs.plot_circuit(c)
    """

    wg1 = straight(length=length_y) if callable(straight) else straight
    wg2 = straight(length=length_y) if callable(straight) else straight
    halfring1 = (
        coupler(length_x=length_x, radius=radius, gap=gap, wg_width=wg_width)
        if callable(coupler)
        else coupler
    )

    halfring2 = (
        coupler(length_x=length_x, radius=radius, gap=gap, wg_width=wg_width)
        if callable(coupler)
        else coupler
    )
    circuit.elements["cb"].pins["o1"] = "o1"
    circuit.elements["cb"].pins["o4"] = "o4"
    circuit.elements["ct"].pins["o4"] = "o2"
    circuit.elements["ct"].pins["o1"] = "o3"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = ring_double()
    plot_circuit(c)
    plt.show()
