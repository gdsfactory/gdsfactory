from __future__ import annotations

import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    coupler_ring: ComponentSpec = coupler_ring_function,
    bend: ComponentSpec = bend_euler,
    cross_section: CrossSectionSpec = "strip",
    **kwargs,
) -> gf.Component:
    """Returns a single ring.

    ring coupler (cb: bottom) connects to two vertical straights (sl: left, sr: right),
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler.
        radius: for the bend and coupler.
        length_x: ring coupler length.
        length_y: vertical straight length.
        coupler_ring: ring coupler spec.
        bend: 90 degrees bend spec.
        cross_section: cross_section spec.
        kwargs: cross_section settings.

    .. code::

          bl-st-br
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x
    """
    gap = gf.snap.snap_to_grid2x(gap)

    c = gf.Component()
    cb = c << gf.get_component(
        coupler_ring,
        bend=bend,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        **kwargs,
    )
    sy = straight(length=length_y, cross_section=cross_section, **kwargs)
    b = gf.get_component(bend, cross_section=cross_section, radius=radius, **kwargs)
    sx = straight(length=length_x, cross_section=cross_section, **kwargs)

    sl = sy.ref()
    sr = sy.ref()
    st = sx.ref()

    if length_y > 0:
        c.add(sl)
        c.add(sr)

    if length_x > 0:
        c.add(st)

    bl = c << b
    br = c << b

    sl.connect(port="o1", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=sl.ports["o2"])

    st.connect(port="o2", destination=bl.ports["o1"])
    br.connect(port="o2", destination=st.ports["o1"])
    sr.connect(port="o1", destination=br.ports["o1"])
    sr.connect(port="o2", destination=cb.ports["o3"])

    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])
    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)
    # c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    # print(c.ports)

    # c = gf.routing.add_fiber_array(ring_single)
    # c = ring_single(cross_section="rib", width=2)
    c = ring_single(length_y=0, length_x=0)
    c.get_netlist()
    c.show(show_ports=True)

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show(show_ports=True)
