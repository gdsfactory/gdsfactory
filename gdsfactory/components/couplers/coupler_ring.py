from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.couplers.coupler import coupler_straight
from gdsfactory.components.couplers.coupler90 import coupler90
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell_with_module_name
def coupler_ring(
    gap: float = 0.2,
    radius: float | None = None,
    length_x: float = 4.0,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float | None = None,
) -> Component:
    r"""Coupler for ring.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends. Default is None, which uses the default radius of the cross_section.
        length_x: length of the parallel coupled straight waveguides.
        bend: 90 degrees bend spec.
        straight: straight spec.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_extension: straight length extension at the end of the coupler bottom ports.

    .. code::

          o2                              o3
          xx                              xx
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────◄──────────────► o4
                                    length_extension
    """
    if radius is None:
        radius = gf.get_cross_section(cross_section).radius
        assert radius is not None, "cross_section must have a radius"

    if length_extension is None:
        length_extension = 3.0 + radius

    c = Component()
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)
    cross_section_bend = cross_section_bend or cross_section

    # define subcells
    coupler90_component = gf.get_component(
        coupler90,
        gap=gap,
        radius=radius,
        bend=bend,
        straight=straight,
        cross_section=cross_section,
        cross_section_bend=cross_section_bend,
        length_straight=length_extension,
    )
    coupler_straight_component = gf.get_component(
        coupler_straight,
        gap=gap,
        length=length_x,
        cross_section=cross_section,
    )

    # add references to subcells
    cbl = c << coupler90_component
    cbr = c << coupler90_component
    cs = c << coupler_straight_component

    # connect references
    cs.connect(port="o4", other=cbr.ports["o1"])
    cbl.connect(port="o2", other=cs.ports["o2"], mirror=True)

    c.add_port("o1", port=cbl.ports["o4"])
    c.add_port("o2", port=cbl.ports["o3"])
    c.add_port("o3", port=cbr.ports["o3"])
    c.add_port("o4", port=cbr.ports["o4"])

    c.add_ports(
        gf.port.select_ports_list(ports=cbl.ports, port_type="electrical"), prefix="cbl"
    )
    c.add_ports(
        gf.port.select_ports_list(ports=cbr.ports, port_type="electrical"), prefix="cbr"
    )
    c.auto_rename_ports()
    c.flatten()
    return c


if __name__ == "__main__":
    c = coupler_ring()
    c.show()
