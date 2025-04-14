from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.typings import CrossSectionSpec, Delta


@gf.cell_with_module_name
def coupler_full(
    coupling_length: float = 40.0,
    dx: Delta = 10.0,
    dy: Delta = 4.8,
    gap: float = 0.5,
    dw: float = 0.1,
    cross_section: CrossSectionSpec = "strip",
    width: float | None = None,
) -> Component:
    """Adiabatic Full coupler.

    Design based on asymmetric adiabatic full
    coupler designs, such as the one reported in 'Integrated Optic Adiabatic
    Devices on Silicon' by Y. Shani, et al (IEEE Journal of Quantum
    Electronics, Vol. 27, No. 3 March 1991).

    1. is the first half of the input S-bend straight where the
    input straights widths taper by +dw and -dw,
    2. is the second half of the S-bend straight with constant,
    unbalanced widths,
    3. is the coupling region where the straights from unbalanced widths to
    balanced widths to reverse polarity unbalanced widths,
    4. is the fixed width straight that curves away from the coupling region,
    5.is the final curve where the straights taper back to the regular width
    specified in the straight template.

    Args:
        coupling_length: Length of the coupling region in um.
        dx: Length of the bend regions in um.
        dy: Port-to-port distance between the bend regions in um.
        gap: Distance between the two straights in um.
        dw: delta width. Top arm tapers to width - dw, bottom to width + dw in um.
        cross_section: cross-section spec.
        width: width of the waveguide. If None, it will use the width of the cross_section.

    """
    c = gf.Component()

    if width:
        x = gf.get_cross_section(cross_section=cross_section, width=width)
    else:
        x = gf.get_cross_section(cross_section=cross_section)
    x_top = x.copy(width=x.width + dw)
    x_bottom = x.copy(width=x.width - dw)

    taper_top = c << gf.components.taper(
        length=coupling_length,
        width1=x_top.width,
        width2=x_bottom.width,
        cross_section=cross_section,
    )

    taper_bottom = c << gf.components.taper(
        length=coupling_length,
        width1=x_bottom.width,
        width2=x_top.width,
        cross_section=cross_section,
    )

    bend_input_top = c << gf.c.bend_s(
        size=(dx, (dy - gap - x_top.width) / 2.0), cross_section=x_top
    )
    bend_input_top.movey((x_top.width + gap) / 2.0)

    bend_input_bottom = c << gf.c.bend_s(
        size=(dx, (-dy + gap + x_bottom.width) / 2.0), cross_section=x_bottom
    )
    bend_input_bottom.movey(-(x_bottom.width + gap) / 2.0)

    taper_top.connect("o1", bend_input_top.ports["o1"])
    taper_bottom.connect("o1", bend_input_bottom.ports["o1"])

    bend_output_top = c << gf.c.bend_s(
        size=(dx, (dy - gap - x_top.width) / 2.0), cross_section=x_bottom
    )

    bend_output_bottom = c << gf.c.bend_s(
        size=(dx, (-dy + gap + x_bottom.width) / 2.0), cross_section=x_top
    )

    bend_output_top.connect("o2", taper_top.ports["o2"], mirror=True)
    bend_output_bottom.connect("o2", taper_bottom.ports["o2"], mirror=True)

    x.add_bbox(c)

    c.add_port("o1", port=bend_input_bottom.ports["o2"])
    c.add_port("o2", port=bend_input_top.ports["o2"])
    c.add_port("o3", port=bend_output_top.ports["o1"])
    c.add_port("o4", port=bend_output_bottom.ports["o1"])
    c.auto_rename_ports()

    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler_full(
        # coupling_length=40,
        # gap=0.2,
        # dw=0.1,
        # cladding_layers=[(111, 0)],
        # cladding_offsets=[3],
    )
    c.show()
    # c.show( )
