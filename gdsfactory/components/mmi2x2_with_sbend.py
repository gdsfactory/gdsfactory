import numpy as np
import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec, CrossSectionSpec
from gdsfactory.components.bend_s import bend_s


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

    xs = gf.get_cross_section(cross_section)
    xs.width = mmi_widths
    c << gf.path.extrude(P, cross_section=xs)

    # Add input and output tapers
    top_input_block = c << gf.components.taper(
        length=1, width1=0.5, width2=0.7, cross_section=cross_section
    ).move((-1, 0.45))
    bottom_input_block = c << gf.components.taper(
        length=1, width1=0.5, width2=0.7, cross_section=cross_section
    ).move((-1, -0.45))

    top_output_block = c << gf.components.taper(
        length=1, width1=0.7, width2=0.5, cross_section=cross_section
    ).move((8, 0.45))
    bottom_output_block = c << gf.components.taper(
        length=1, width1=0.7, width2=0.5, cross_section=cross_section
    ).move((8, -0.45))

    if with_sbend:
        sbend = gf.get_component(s_bend, cross_section=cross_section)

        top_input_sbend_ref = c << sbend
        top_input_sbend_ref.mirror([0, 1])
        bottom_input_sbend_ref = c << sbend
        top_output_sbend_ref = c << sbend
        bottom_output_sbend_ref = c << sbend
        bottom_output_sbend_ref.mirror([0, 1])

        top_input_sbend_ref.connect("o1", destination=top_input_block.ports["o1"])
        bottom_input_sbend_ref.connect("o1", destination=bottom_input_block.ports["o1"])
        top_output_sbend_ref.connect("o1", destination=top_output_block.ports["o2"])
        bottom_output_sbend_ref.connect(
            "o1", destination=bottom_output_block.ports["o2"]
        )

        c.add_port("o1", port=bottom_input_sbend_ref.ports["o2"])
        c.add_port("o2", port=top_input_sbend_ref.ports["o2"])
        c.add_port("o3", port=top_output_sbend_ref.ports["o2"])
        c.add_port("o4", port=bottom_output_sbend_ref.ports["o2"])

    else:
        c.add_port("o2", port=top_input_block.ports["o1"])
        c.add_port("o1", port=bottom_input_block.ports["o1"])
        c.add_port("o3", port=top_output_block.ports["o2"])
        c.add_port("o4", port=bottom_output_block.ports["o2"])
    return c


if __name__ == "__main__":
    c = mmi2x2_with_sbend(
        with_sbend=True,
        cross_section=dict(cross_section="strip", settings=dict(layer=(2, 0))),
    )
    # c = mmi2x2_with_sbend(with_sbend=False)
    c.show(show_ports=True)
