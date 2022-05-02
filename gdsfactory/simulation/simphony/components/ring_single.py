from simphony.netlist import Subcircuit

from gdsfactory.simulation.simphony.components.bend_circular import bend_circular
from gdsfactory.simulation.simphony.components.coupler_ring import coupler_ring
from gdsfactory.simulation.simphony.components.straight import straight
from gdsfactory.simulation.simphony.plot_circuit import plot_circuit


def ring_single(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    radius=5,
    length_y=2,
    coupler=coupler_ring,
    straight=straight,
    bend=bend_circular,
):
    r"""Return single bus ring Model made of a ring coupler (cb: bottom).

    connected with:
    - 2 vertical straights (wl: left, wr: right)
    - 2 bend90 straights (bl: left, br: right)
    - 1 straight at the top (wt)

    FIXME! Sparameters are zero

    .. code::

              wt (top)
              length_x
             /         \
            /           \
           |             |
           o1           o1 ___
                            |
          wl            wr  | length_y
                           _|_
           o2            o3
           |             |
            \           /
             \         /
           ---=========---
        o1 o1 length_x  o4 o2



    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.ring_single(width=0.5, gap=0.2, length_x=4, radius=5, length_y=2)
      c.plot()


    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.ring_single()
        gs.plot_circuit(c)
    """
    straight = (
        straight(width=wg_width, length=length_y) if callable(straight) else straight
    )
    bend = bend(width=wg_width, radius=radius) if callable(bend) else bend
    coupler = (
        coupler(length_x=length_x, radius=radius, gap=gap, wg_width=wg_width)
        if callable(coupler)
        else coupler
    )

    # Create the circuit, add all individual instances
    circuit = Subcircuit("ring_double")
    circuit.add(
        [
            (bend, "bl"),
            (bend, "br"),
            (coupler, "cb"),
            (straight, "wl"),
            (straight, "wr"),
            (straight, "wt"),
        ]
    )

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("cb", "o2", "wl", "o2"),
            ("wl", "o1", "bl", "o2"),
            ("bl", "o1", "wt", "o1"),
            ("wt", "o2", "br", "o1"),
            ("br", "o2", "wr", "o2"),
            ("wr", "o1", "cb", "o3"),
        ]
    )
    circuit.elements["cb"].pins["o1"] = "o1"
    circuit.elements["cb"].pins["o4"] = "o2"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = ring_single()
    plot_circuit(c)
    plt.show()
