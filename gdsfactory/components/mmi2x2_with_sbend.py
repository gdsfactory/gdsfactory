import numpy as np

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mmi2x2_with_sbend(
    with_sbend: bool = True,
    s_bend: ComponentSpec = bend_s,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    """Returns mmi2x2 for Cband.

    C_band 2x2MMI in 220nm thick silicon
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-23-28957&id=376719

    Args:
        with_sbend: add sbend.
        s_bend: S-bend spec.
        cross_section: spec.
    """

    def mmi_widths(t):
        # Note: Custom width/offset functions MUST be vectorizable--you must be able
        # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
        widths = np.array([2 * 0.7 + 0.2, 1.48, 1.48, 1.48, 1.6])
        return widths

    c = gf.Component()

    P = gf.path.straight(length=2 * 2.4 + 2 * 1.6, npoints=5)

    xs = gf.get_cross_section(cross_section, add_pins=None)
    xs.width = mmi_widths
    ref = c << gf.path.extrude(P, cross_section=xs)

    # Add input and output tapers
    taper = gf.components.taper(
        length=1, width1=0.5, width2=0.7, cross_section=cross_section, add_pins=None
    )
    topl_taper = c << taper
    topl_taper.move((-1, 0.45))
    botl_taper = c << taper
    botl_taper.move((-1, -0.45))

    topr_taper = c << taper
    topr_taper.mirror()
    topr_taper.move((9, 0.45))

    botr_taper = c << taper
    botr_taper.mirror()
    botr_taper.move((9, -0.45))

    if with_sbend:
        sbend = gf.get_component(s_bend, cross_section=cross_section, add_pins=None)

        topl_sbend = c << sbend
        topl_sbend.mirror([0, 1])
        botl_sbend = c << sbend
        topr_sbend = c << sbend
        botr_sbend = c << sbend
        botr_sbend.mirror([0, 1])

        topl_sbend.connect("o1", destination=topl_taper.ports["o1"])
        botl_sbend.connect("o1", destination=botl_taper.ports["o1"])
        topr_sbend.connect("o1", destination=topr_taper.ports["o1"])
        botr_sbend.connect("o1", destination=botr_taper.ports["o1"])

        c.add_port("o1", port=botl_sbend.ports["o2"])
        c.add_port("o2", port=topl_sbend.ports["o2"])
        c.add_port("o3", port=topr_sbend.ports["o2"])
        c.add_port("o4", port=botr_sbend.ports["o2"])

        c.absorb(topr_sbend)
        c.absorb(topl_sbend)
        c.absorb(botr_sbend)
        c.absorb(botl_sbend)
    else:
        c.add_port("o2", port=topl_taper.ports["o1"])
        c.add_port("o1", port=botl_taper.ports["o1"])
        c.add_port("o3", port=topr_taper.ports["o1"])
        c.add_port("o4", port=botr_taper.ports["o1"])

    xs = gf.get_cross_section(cross_section)
    if xs.add_pins:
        c = xs.add_pins(c)

    c.absorb(ref)

    c.absorb(topr_taper)
    c.absorb(topl_taper)
    c.absorb(botr_taper)
    c.absorb(botl_taper)
    return c


if __name__ == "__main__":
    # c = mmi2x2_with_sbend(
    #     with_sbend=True,
    #     cross_section=dict(cross_section="strip", settings=dict(layer=(2, 0))),
    # )
    c = mmi2x2_with_sbend(with_sbend=False)
    c.show(show_ports=True)
