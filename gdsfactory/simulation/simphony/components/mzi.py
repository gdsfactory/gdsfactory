from typing import Callable, Optional

from gdsfactory.simulation.simphony.components.mmi1x2 import mmi1x2
from gdsfactory.simulation.simphony.components.straight import (
    straight as straight_function,
)


def mzi(
    delta_length: float = 10.0,
    length_y: float = 4.0,
    length_x: float = 0.1,
    splitter: Callable = mmi1x2,
    combiner: Optional[Callable] = None,
    straight_top: Callable = straight_function,
    straight_bot: Callable = straight_function,
    port_name_splitter_w0: str = "o1",
    port_name_splitter_e1: str = "o2",
    port_name_splitter_e0: str = "o3",
    port_name_combiner_w0: str = "o1",
    port_name_combiner_e1: str = "o2",
    port_name_combiner_e0: str = "o3",
):
    """Returns Mzi circuit model.

    Args:
        delta_length: bottom arm vertical extra length.
        length_y: vertical length for both and top arms.
        length_x: horizontal length.
        splitter: model function for combiner.
        combiner: model function for combiner.
        wg: straight model function.

    .. code::


                   __Lx__
                  |      |
                  Ly     Lyr
                  |      |
         splitter=|      |==combiner
                  |      |
                  Ly     Lyr
                  |      |
                 DL/2   DL/2
                  |      |
                  |__Lx__|



    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.mzi(delta_length=10)
      c.plot()


    .. plot::
        :include-source:

        import gdsfactory.simulation.simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.mzi()
        gs.plot_circuit(c)

    """
    combiner = combiner or splitter
    splitter = splitter() if callable(splitter) else splitter
    combiner = combiner() if callable(combiner) else combiner

    wg_short = straight_top(length=2 * length_y + length_x)
    wg_long = straight_bot(length=2 * length_y + delta_length + length_x)

    splitter[port_name_combiner_e0].connect(wg_long["o1"])
    splitter[port_name_combiner_e1].connect(wg_short["o1"])
    combiner[port_name_combiner_e0].connect(wg_long["o2"])
    combiner[port_name_combiner_e1].connect(wg_short["o2"])

    splitter[port_name_splitter_w0].rename("o1")
    combiner[port_name_combiner_w0].rename("o2")

    return splitter.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_circuit import plot_circuit

    c = mzi()
    plot_circuit(c)
    plt.show()
