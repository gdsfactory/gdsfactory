from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import ComponentSpec, CrossSectionSpec


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    bend: ComponentSpec = "bend_euler",
    straight: ComponentSpec = "straight",
    coupler_ring: ComponentSpec = "coupler_ring",
    cross_section: CrossSectionSpec = "strip",
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
        straight: straight spec.
        coupler_ring: ring coupler spec.
        cross_section: cross_section spec.

    .. code::

                    xxxxxxxxxxxxx
                xxxxx           xxxx
              xxx                   xxx
            xxx                       xxx
           xx                           xxx
           x                             xxx
          xx                              xx▲
          xx                              xx│length_y
          xx                              xx▼
          xx                             xx
           xx          length_x          x
            xx     ◄───────────────►    x
             xx                       xxx
               xx                   xxx
                xxx──────▲─────────xxx
                         │gap
                 o1──────▼─────────o2
    """
    if length_y < 0:
        raise ValueError(f"length_y={length_y} must be >= 0")

    if length_x < 0:
        raise ValueError(f"length_x={length_x} must be >= 0")

    c = gf.Component()
    cb = c << gf.get_component(
        coupler_ring,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        bend=bend,
        straight=straight,
    )
    sy = gf.get_component(straight, length=length_y, cross_section=cross_section)
    b = gf.get_component(bend, cross_section=cross_section, radius=radius)
    sx = gf.get_component(straight, length=length_x, cross_section=cross_section)

    sl = c << sy
    sr = c << sy
    st = c << sx
    bl = c << b
    br = c << b

    sl.connect(port="o1", other=cb.ports["o2"])
    bl.connect(port="o2", other=sl.ports["o2"])
    st.connect(port="o2", other=bl.ports["o1"])
    br.connect(port="o2", other=st.ports["o1"])
    sr.connect(port="o1", other=br.ports["o1"])
    sr.connect(port="o2", other=cb.ports["o3"])

    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])
    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)
    # c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    c = ring_single(radius=5, gap=0.111, bend="bend_circular", length_x=0, length_y=0)
    # print(c.ports)

    # c = gf.routing.add_fiber_array(ring_single)
    # c = ring_single(cross_section="rib", width=2)
    # c = ring_single(length_y=0, length_x=0)
    # c.get_netlist()
    c.show()

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show( )
