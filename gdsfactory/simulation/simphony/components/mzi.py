from typing import Callable, Optional

from simphony.netlist import Subcircuit

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
    """Mzi circuit model.

    Args:
        delta_length: bottom arm vertical extra length
        length_y: vertical length for both and top arms
        length_x: horizontal length
        splitter: model function for combiner
        combiner: model function for combiner
        wg: straight model function

    Return: mzi circuit model

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

        import gdsfactory.simulation simphony as gs
        import gdsfactory.simulation.simphony.components as gc

        c = gc.mzi()
        gs.plot_circuit(c)

    """
    combiner = combiner or splitter
    splitter = splitter() if callable(splitter) else splitter
    combiner = combiner() if callable(combiner) else combiner

    wg_short = straight_top(length=2 * length_y + length_x)
    wg_long = straight_bot(length=2 * length_y + delta_length + length_x)

    # Create the circuit, add all individual instances
    circuit = Subcircuit("mzi")
    circuit.add(
        [
            (splitter, "splitter"),
            (combiner, "recombiner"),
            (wg_long, "wg_long"),
            (wg_short, "wg_short"),
        ]
    )

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("splitter", port_name_splitter_e0, "wg_long", "o1"),
            ("splitter", port_name_splitter_e1, "wg_short", "o1"),
            ("recombiner", port_name_combiner_e0, "wg_long", "o2"),
            ("recombiner", port_name_combiner_e1, "wg_short", "o2"),
        ]
    )
    circuit.elements["splitter"].pins[port_name_splitter_w0] = "o1"
    circuit.elements["recombiner"].pins[port_name_combiner_w0] = "o2"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony.plot_circuit import plot_circuit

    c = mzi()
    plot_circuit(c)
    plt.show()
