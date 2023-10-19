from __future__ import annotations

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler90 import coupler90
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.straight import straight
from gdsfactory.typings import (
    ComponentFactory,
    ComponentSpec,
    Coordinates,
    CrossSectionSpec,
    LayerSpecs,
)


@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    coupler90: ComponentFactory = coupler90,
    bend: ComponentSpec = bend_euler,
    coupler_straight: ComponentFactory = coupler_straight,
    cross_section: CrossSectionSpec = "xs_sc",
    cross_section_bend: CrossSectionSpec | None = None,
    length_extension: float = 3,
    add_bbox: callable | None = None,
) -> Component:
    r"""Coupler for ring.

    Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        coupler90: straight coupled to a 90deg bend.
        bend: bend spec.
        coupler_straight: two parallel coupled straight waveguides.
        cross_section: cross_section spec.
        cross_section_bend: optional bend cross_section spec.
        length_extension: for the ports.

    .. code::

          o2            o3
           |             |
            \           /
             \         /
           ---=========---
        o1    length_x   o4

    """
    c = Component()
    gap = gf.snap.snap_to_grid(gap, grid_factor=2)
    xs = gf.get_cross_section(cross_section)
    xs_no_pins = xs.copy(add_pins_function_name=None)

    cross_section_bend = cross_section_bend or xs
    xs_bend = gf.get_cross_section(cross_section_bend)
    xs_bend = xs_bend.copy(radius=radius, add_pins_function_name=None)

    # define subcells
    coupler90_component = coupler90(
        gap=gap,
        radius=radius,
        bend=bend,
        cross_section=xs_no_pins,
        cross_section_bend=xs_bend,
    )
    coupler_straight_component = coupler_straight(
        gap=gap,
        length=length_x,
        cross_section=xs_no_pins,
    )

    # add references to subcells
    cbl = c << coupler90_component
    cbr = c << coupler90_component
    cs = coupler_straight_component.ref()

    if length_x > 0:
        c.add(cs)

    # connect references
    y = coupler90_component.y
    cs.connect(port="o4", destination=cbr.ports["o1"])
    cbl.mirror(p1=(0, y), p2=(1, y))
    cbl.connect(port="o2", destination=cs.ports["o2"])

    s = straight(length=length_extension, cross_section=xs_no_pins)

    s1 = c << s
    s2 = c << s

    s1.connect("o2", cbl["o4"])
    s2.connect("o1", cbr["o4"])

    c.add_port("o1", port=s1["o1"])
    c.add_port("o2", port=cbl["o3"])
    c.add_port("o3", port=cbr["o3"])
    c.add_port("o4", port=s2["o2"])

    c.add_ports(cbl.get_ports_list(port_type="electrical"), prefix="cbl")
    c.add_ports(cbr.get_ports_list(port_type="electrical"), prefix="cbr")
    c.auto_rename_ports()
    if add_bbox:
        add_bbox(c)
    xs.add_pins(c)
    return c


@gf.cell
def coupler_ring_point(
    coupler_ring: ComponentFactory = coupler_ring,
    open_layers: LayerSpecs = None,
    open_sizes: Coordinates | None = None,
    **kwargs,
) -> Component:
    """Coupler ring that removes some layers at the coupling region.

    This allows short interaction lengths (point couplers).

    Args:
        coupler_ring: coupler_ring component to process.
        open_layers: layers to perform the boolean operations on.
        open_sizes: sizes of the boxes to use to remove layers, centered at bus center.

    Keyword Args:
        gap: spacing between parallel coupled straight waveguides.
        radius: of the bends.
        length_x: length of the parallel coupled straight waveguides.
        coupler90: straight coupled to a 90deg bend.
        bend: bend spec.
        coupler_straight: two parallel coupled straight waveguides.
        cross_section: cross_section spec.
        length_extension: for the ports.
    """
    c = gf.Component()

    coupler_ring_component = coupler_ring(**kwargs)
    open_layers = open_layers or []
    open_sizes = open_sizes or []

    open_layers_tuples = [gf.get_layer(open_layer) for open_layer in open_layers]
    untouched_layers = list(
        set(coupler_ring_component.get_layers()) - set(open_layers_tuples)
    )

    for layer, size in zip(open_layers, open_sizes):
        subcomponent = coupler_ring_component.extract(layers=[layer])
        rectangle = gf.components.rectangle(size=size, layer=layer, centered=True)
        _ = c << gf.geometry.boolean(subcomponent, rectangle, "A-B", layer=layer)

    coupler_ref = c << coupler_ring_component.extract(layers=untouched_layers)
    c.add_ports(coupler_ring_component.get_ports_list())
    c.absorb(coupler_ref)
    return c


if __name__ == "__main__":
    c = coupler_ring(cross_section_bend="xs_sc_heater_metal")
    c.show(show_ports=False)
