from simphony.library import siepic
from simphony.netlist import Subcircuit


def ring_double_siepic(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    bend_radius=5,
    length_y=2,
    coupler=siepic.ebeam_dc_halfring_straight,
    straight=siepic.ebeam_wg_integral_1550,
    terminator=siepic.ebeam_terminator_te1550,
):
    r"""Return double bus ring made of two couplers (ct: top, cb: bottom).

    connected with two vertical straights (wyl: left, wyr: right)

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x


     drop   n1 _        _ n3 cdrop
                \______/
                 ______
     in     n2 _/      \_n4
               |        |
            n1 |        | n3
                \______/
                 ______
     in     n2 _/      \_n4 output


    """
    straight = straight() if callable(straight) else straight
    coupler = coupler() if callable(coupler) else coupler

    # Create the circuit, add all individual instances
    circuit = Subcircuit("mzi")
    circuit.add([(coupler, "ct"), (coupler, "cb"), (straight, "wl"), (straight, "wr")])

    # Circuits can be connected using the elements' string names:
    circuit.connect_many(
        [
            ("cb", "n1", "wl", "n1"),
            ("wl", "n2", "ct", "n2"),
            ("ct", "n4", "wr", "n1"),
            ("wr", "n2", "cb", "n3"),
        ]
    )
    circuit.elements["cb"].pins["n2"] = "input"
    circuit.elements["cb"].pins["n4"] = "output"
    circuit.elements["ct"].pins["n1"] = "drop"
    circuit.elements["ct"].pins["n3"] = "cdrop"
    return circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulationsimphony import plot_circuit

    c = ring_double_siepic()
    plot_circuit(c)
    plt.show()
