"""
Lets define the references from a component and then connect them together

"""

import pp


@pp.autoname
def ring(
    coupler90=pp.c.coupler90,
    coupler_straight=pp.c.coupler_straight,
    waveguide=pp.c.waveguide,
    bend=pp.c.bend_circular,
    length_y=2.0,
    length_x=4.0,
    gap=0.2,
    wg_width=0.5,
):
    """ single bus ring
    """
    c = pp.Component()

    # define subcells
    coupler90 = pp.call_if_func(coupler90, gap=gap, width=wg_width)
    waveguide_x = pp.call_if_func(waveguide, length=length_x, width=wg_width)
    waveguide_y = pp.call_if_func(waveguide, length=length_y, width=wg_width)
    bend = pp.call_if_func(bend, width=wg_width)
    coupler_straight = pp.call_if_func(
        coupler_straight, gap=gap, length=length_x, width=wg_width
    )

    # add references to subcells
    cbl = c << coupler90
    cbr = c << coupler90
    cs = c << coupler_straight
    wyl = c << waveguide_y
    wyr = c << waveguide_y
    wx = c << waveguide_x
    btl = c << bend
    btr = c << bend

    # connect references
    wyr.connect(port="E0", destination=cbr.ports["N0"])
    cs.connect(port="E0", destination=cbr.ports["W0"])

    cbl.reflect(p1=(0, coupler90.y), p2=(1, coupler90.y))
    cbl.connect(port="W0", destination=cs.ports["W0"])
    wyl.connect(port="E0", destination=cbl.ports["N0"])

    btl.connect(port="N0", destination=wyl.ports["W0"])
    btr.connect(port="W0", destination=wyr.ports["W0"])
    wx.connect(port="W0", destination=btl.ports["W0"])
    return c


if __name__ == "__main__":
    c = ring(wg_width=0.4)
    pp.write_gds(c, "ring.gds")
    pp.show(c)
