from __future__ import annotations

from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler90 import coupler90
from gdsfactory.components.coupler_straight import coupler_straight
from gdsfactory.components.straight import straight
from gdsfactory.typings import (
    ComponentSpec,
    CrossSectionSpec,
    LayerSpecs,
    Coordinates,
    ComponentFactory,
)


@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 5.0,
    length_x: float = 4.0,
    coupler90: ComponentSpec = coupler90,
    bend: ComponentSpec = bend_euler,
    coupler_straight: ComponentSpec = coupler_straight,
    cross_section: CrossSectionSpec = "strip",
    bend_cross_section: Optional[CrossSectionSpec] = None,
    length_extension: float = 3,
    **kwargs,
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
        bend_cross_section: optional bend cross_section spec.
        length_extension: for the ports.
        kwargs: cross_section settings for bend and coupler.

    .. code::

           2             3
           |             |
            \           /
             \         /
           ---=========---
         1    length_x    4

    """
    c = Component()
    gap = gf.snap.snap_to_grid(gap, nm=2)

    # define subcells
    coupler90_component = gf.get_component(
        coupler90,
        gap=gap,
        radius=radius,
        bend=bend,
        cross_section=cross_section,
        bend_cross_section=bend_cross_section,
        **kwargs,
    )
    coupler_straight_component = gf.get_component(
        coupler_straight,
        gap=gap,
        length=length_x,
        cross_section=cross_section,
        **kwargs,
    )

    # add references to subcells
    cbl = c << coupler90_component
    cbr = c << coupler90_component
    cs = c << coupler_straight_component

    # connect references
    y = coupler90_component.y
    cs.connect(port="o4", destination=cbr.ports["o1"])
    cbl.mirror(p1=(0, y), p2=(1, y))
    cbl.connect(port="o2", destination=cs.ports["o2"])

    s = straight(length=length_extension, cross_section=cross_section, **kwargs)

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
    return c


@gf.cell
def coupler_ring_point(
    coupler_ring: ComponentFactory = coupler_ring,
    open_layers: LayerSpecs = None,
    open_sizes: Optional[Coordinates] = None,
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
        bend_cross_section: optional bend cross_section spec.
        length_extension: for the ports.
        kwargs: cross_section settings for bend and coupler.
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
        c << gf.geometry.boolean(subcomponent, rectangle, "A-B", layer=layer)

    coupler_ref = c << coupler_ring_component.extract(layers=untouched_layers)

    c.add_ports(coupler_ring_component.get_ports_list())
    c.absorb(coupler_ref)

    return c


if __name__ == "__main__":
    # c = coupler_ring()
    # c = coupler_ring(width=1, layer=(2, 0), length_x=20)
    # c = coupler_ring(
    #     cross_section="strip_heater_metal",
    #     length_x=0,
    #     bend=gf.components.bend_circular,
    # )
    # from functools import partial

    # c = partial(
    #     coupler_ring,
    #     cross_section="strip_heater_metal",
    #     length_x=0,
    #     bend=gf.components.bend_circular,
    # )
    # c = coupler_ring_point(c, open_layers=("HEATER",), open_sizes=((5, 7),))

    c = coupler_ring_point()

    # c = gf.Component()
    # c1 = coupler_ring(cladding_layers=[(111, 0)], cladding_offsets=[0.5])
    # d = 0.8
    # c2 = gf.geometry.offset(c1, distance=+d, layer=(111, 0))
    # c3 = gf.geometry.offset(c2, distance=-d, layer=(111, 0))
    # c << c1
    # c << c3
    c.show(show_ports=True)
