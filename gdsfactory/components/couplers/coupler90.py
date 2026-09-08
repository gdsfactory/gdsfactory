from __future__ import annotations

__all__ = ["coupler90", "coupler90circular"]

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.cross_section import ExtrusionSpec
from gdsfactory.typings import ComponentSpec, CrossSectionSpec

from .._schematic import coupler_schematic


@gf.cell_with_module_name(schematic_function=coupler_schematic, tags=["couplers"])
def coupler90(
    gap: float = 0.2,
    radius: float | None = None,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    cross_section: CrossSectionSpec = "strip",
    cross_section_bend: CrossSectionSpec | None = None,
    length_straight: float | None = None,
    extrusion_spec: ExtrusionSpec | None = None,
) -> Component:
    r"""Straight coupled to a bend.

    Args:
        gap: um.
        radius: um.
        straight: for straight.
        bend: bend spec.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_straight: optional length of the straight waveguide.
        extrusion_spec: optional extrusion metadata for the bend cross-section.

    ```text
            o3
             |
            /
           /
       o2_/
       o1___o4
    ```

    """
    c = Component()
    x = gf.get_cross_section(cross_section)
    xs_bend = cross_section_bend or cross_section

    bend_kwargs = {"radius": radius, "cross_section": xs_bend}
    if extrusion_spec is not None:
        bend_kwargs["extrusion_spec"] = extrusion_spec
    bend90 = gf.get_component(bend, **bend_kwargs)
    bend_ref = c << bend90
    bend90_ports = bend_ref.ports.filter(port_type="optical")

    if length_straight is None:
        length_straight = bend90_ports[1].center[0] - bend90_ports[0].center[0]

    straight_component = gf.get_component(
        straight,
        cross_section=cross_section,
        length=length_straight,
    )
    wg_ref = c << straight_component
    width = x.width

    pbw = bend90_ports[0]
    bend_ref.movey(pbw.y + gap + width)
    c.add_ports(wg_ref.ports, prefix="wg")
    c.add_ports(bend_ref.ports, prefix="bend")
    c.auto_rename_ports()
    return c


coupler90circular = partial(coupler90, bend="bend_circular")
