from simphony.libraries import siepic


def ring_double_siepic(
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    bend_radius=5,
    length_y=2,
    coupler=siepic.DirectionalCoupler,
    straight=siepic.Waveguide,
    terminator=siepic.Terminator,
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
    ct = cb = coupler
    wl = wr = straight

    cb.rename_pins("n2", "n1", "n4", "n3")
    ct.rename_pins("n2", "n1", "n4", "n3")
    wl.rename_pins("n1", "n2")
    wr.rename_pins("n1", "n2")

    ct["n2"].connect(wl["n2"])
    ct["n4"].connect(wr["n1"])
    cb["n1"].connect(wl["n1"])
    cb["n3"].connect(wr["n2"])

    cb["n2"].rename("input")
    cb["n4"].rename("output")
    ct["n1"].rename("drop")
    ct["n3"].rename("cdrop")

    return cb.circuit.to_subcircuit()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from gdsfactory.simulation.simphony import plot_circuit

    c = ring_double_siepic()
    plot_circuit(c, pin_in="input", pins_out=("output",))
    plt.show()
