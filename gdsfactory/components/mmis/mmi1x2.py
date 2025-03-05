from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.tapers.taper import taper as taper_function
from gdsfactory.components.waveguides.straight import straight as straight_function
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def mmi1x2(
    width: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    taper: ComponentSpec = taper_function,
    straight: ComponentSpec = straight_function,
    cross_section: CrossSectionSpec = "strip",
) -> Component:
    r"""1x2 MultiMode Interferometer (MMI).

    Args:
        width: input and output straight width. Defaults to cross_section width.
        width_taper: interface between input straights and mmi region.
        length_taper: into the mmi region.
        length_mmi: in x direction.
        width_mmi: in y direction.
        gap_mmi:  gap between tapered wg.
        taper: taper function.
        straight: straight function.
        cross_section: specification (CrossSection, string or dict).


    .. code::

               length_mmi
                <------>
                ________
               |        |
               |         \__
               |          __  o2
            __/          /_ _ _ _
         o1 __          | _ _ _ _| gap_mmi
              \          \__
               |          __  o3
               |         /
               |________|

             <->
        length_taper

    """
    c = Component()
    gap_mmi = gf.snap.snap_to_grid(gap_mmi, grid_factor=2)
    x = gf.get_cross_section(cross_section)
    xs_mmi = gf.get_cross_section(cross_section, width=width_mmi)
    width = width or x.width

    _taper = gf.get_component(
        taper,
        length=length_taper,
        width1=width,
        width2=width_taper,
        cross_section=cross_section,
    )

    a = gap_mmi / 2 + width_taper / 2
    _ = c << gf.get_component(straight, length=length_mmi, cross_section=xs_mmi)

    temp_component = Component()

    ports = [
        temp_component.add_port(
            name="o1",
            orientation=180,
            center=(0, 0),
            width=width_taper,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        ),
        temp_component.add_port(
            name="o2",
            orientation=0,
            center=(+length_mmi, +a),
            width=width_taper,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        ),
        temp_component.add_port(
            name="o3",
            orientation=0,
            center=(+length_mmi, -a),
            width=width_taper,
            layer=gf.get_layer(x.layer),
            cross_section=x,
        ),
    ]

    for port in ports:
        taper_ref = c << _taper
        taper_ref.connect(port="o2", other=port, allow_width_mismatch=True)
        c.add_port(name=port.name, port=taper_ref.ports["o1"])

    c.flatten()
    return c


if __name__ == "__main__":
    c = mmi1x2(cross_section="rib")
    c.show()
