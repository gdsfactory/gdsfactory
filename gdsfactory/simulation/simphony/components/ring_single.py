from simphony.models import Subcircuit

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
) -> Subcircuit:
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
    wg1 = straight
    wg2 = straight
    bend.connect(wg1)
    bend.connect(wg2)
    coupler.multiconnect(wg1["o2"], wg2["o2"])

    return coupler.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = ring_single()
    plot_circuit(c, pins_out=("o4",))
    plt.show()
