from simphony.libraries.sipann import Racetrack
from simphony.models import Subcircuit

from gdsfactory.simulation.simphony.plot_circuit import plot_circuit


def ring_single(
    wg_width: float = 0.5,
    gap: float = 0.2,
    length_x: float = 4,
    radius: float = 5,
    length_y: float = 2,
) -> Subcircuit:
    r"""Return single bus ring Model made of a ring coupler (cb: bottom).

    connected with:
    - 2 vertical straights (wl: left, wr: right)
    - 2 bend90 straights (bl: left, br: right)
    - 1 straight at the top (wt)

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
    wg_width *= 1e-6
    gap *= 1e-6
    length_x *= 1e-6
    radius *= 1e-6
    length_y *= 1e-6

    racetrack = Racetrack(wg_width, 0.22e-6, radius, gap, length_x)

    racetrack.rename_pins("o1", "o2")
    return racetrack.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = ring_single()
    plot_circuit(c)
    plt.show()
