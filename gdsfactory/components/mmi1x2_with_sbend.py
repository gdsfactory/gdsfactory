import numpy as np
from scipy.interpolate import interp1d

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec
from gdsfactory.components.bend_s import bend_s


@gf.cell
def mmi1x2_with_sbend(
    with_sbend: bool = False,
    s_bend: ComponentSpec = bend_s,
) -> Component:
    """Returns 1x2 splitter

    From this paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-1-1310&id=248418
    """

    def mmi_widths(t):
        # Note: Custom width/offset functions MUST be vectorizable--you must be able
        # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
        widths = np.array(
            [0.5, 0.5, 0.6, 0.7, 0.9, 1.26, 1.4, 1.4, 1.4, 1.4, 1.31, 1.2, 1.2]
        )
        xold = np.linspace(0, 1, num=len(widths))
        xnew = np.linspace(0, 1, num=100)
        f = interp1d(xold, widths, kind="cubic")
        return f(xnew)

    c = gf.Component()

    P = gf.path.straight(length=2, npoints=100)
    X = gf.CrossSection(width=mmi_widths, offset=0, layer="WG")
    c << gf.path.extrude(P, cross_section=X)

    # Add "stub" straight sections for ports
    input_port_ref = c << gf.components.straight(length=0.25, cross_section="strip")
    input_port_ref.center = (-0.125, 0)
    top_output_port_ref = c << gf.components.straight(
        length=0.25, cross_section="strip"
    )
    top_output_port_ref.center = (2.125, 0.35)
    bottom_output_port_ref = c << gf.components.straight(
        length=0.25, cross_section="strip"
    )
    bottom_output_port_ref.center = (2.125, -0.35)
    if with_sbend:
        top_sbend = c << s_bend()
        bot_sbend = c << s_bend()
        bot_sbend.mirror([1, 0])
        top_sbend.connect("o1", destination=top_output_port_ref.ports["o2"])
        bot_sbend.connect("o1", destination=bottom_output_port_ref.ports["o2"])
        c.add_port("o1", port=input_port_ref.ports["o1"])
        c.add_port("o2", port=top_sbend.ports["o2"])
        c.add_port("o3", port=bot_sbend.ports["o2"])

    else:
        c.add_port("o1", port=input_port_ref.ports["o1"])
        c.add_port("o2", port=top_output_port_ref.ports["o2"])
        c.add_port("o3", port=bottom_output_port_ref.ports["o2"])

    return c


if __name__ == "__main__":
    c = mmi1x2_with_sbend(with_sbend=False)
    c.show(show_ports=True)
