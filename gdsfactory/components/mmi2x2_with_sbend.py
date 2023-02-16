import numpy as np
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec
from gdsfactory.components.bend_s import bend_s


@gf.cell
def mmi2x2_with_sbend(
    with_sbend: bool = False,
    s_bend: ComponentSpec = bend_s,
) -> Component:
    """Returns mmi2x2 for Cband.

    A C_band 2x2MMI in 220nm thick silicon from
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-25-23-28957&id=376719
    """

    def mmi_widths(t):
        # Note: Custom width/offset functions MUST be vectorizable--you must be able
        # to call them with an array input like my_custom_width_fun([0, 0.1, 0.2, 0.3, 0.4])
        widths = np.array([2 * 0.7 + 0.2, 1.48, 1.48, 1.48, 1.6])
        return widths

    c = gf.Component()

    P = gf.path.straight(length=2 * 2.4 + 2 * 1.6, npoints=5)
    X = gf.CrossSection(width=mmi_widths, offset=0, layer="WG")
    c << gf.path.extrude(P, cross_section=X)

    # Add input and output tapers
    top_input_block = c << gf.components.taper(
        length=1, width1=0.5, width2=0.7, layer="WG"
    ).move((-1, 0.45))
    bottom_input_block = c << gf.components.taper(
        length=1, width1=0.5, width2=0.7, layer="WG"
    ).move((-1, -0.45))

    top_output_block = c << gf.components.taper(
        length=1, width1=0.7, width2=0.5, layer="WG"
    ).move((8, 0.45))
    bottom_output_block = c << gf.components.taper(
        length=1, width1=0.7, width2=0.5, layer="WG"
    ).move((8, -0.45))

    if with_sbend:
        top_input_sbend_ref = c << s_bend()
        top_input_sbend_ref.mirror([0, 1])
        bottom_input_sbend_ref = c << s_bend()
        top_output_sbend_ref = c << s_bend()
        bottom_output_sbend_ref = c << s_bend()
        bottom_output_sbend_ref.mirror([0, 1])

        top_input_sbend_ref.connect("o1", destination=top_input_block.ports["o1"])
        bottom_input_sbend_ref.connect("o1", destination=bottom_input_block.ports["o1"])
        top_output_sbend_ref.connect("o1", destination=top_output_block.ports["o2"])
        bottom_output_sbend_ref.connect(
            "o1", destination=bottom_output_block.ports["o2"]
        )

        c.add_port("o1", port=top_input_sbend_ref.ports["o2"])
        c.add_port("o2", port=top_output_sbend_ref.ports["o2"])
        c.add_port("o3", port=bottom_output_sbend_ref.ports["o2"])
        c.add_port("o4", port=bottom_input_sbend_ref.ports["o2"])

    else:
        c.add_port("o1", port=top_input_block.ports["o1"])
        c.add_port("o4", port=bottom_input_block.ports["o1"])
        c.add_port("o2", port=top_output_block.ports["o2"])
        c.add_port("o3", port=bottom_output_block.ports["o2"])
    return c


if __name__ == "__main__":
    # c = mmi2x2_with_sbend(with_sbend=True)
    c = mmi2x2_with_sbend(with_sbend=False)
    c.show(show_ports=True)
