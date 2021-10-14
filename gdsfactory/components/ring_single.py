import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as _coupler_ring
from gdsfactory.components.straight import straight as _straight


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    coupler_ring: gf.types.ComponentFactory = _coupler_ring,
    straight: gf.types.ComponentFactory = _straight,
    bend: gf.types.ComponentFactory = bend_euler,
    cross_section: gf.types.CrossSectionFactory = gf.cross_section.strip,
    **kwargs
) -> gf.Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical straights (sl: left, sr: right)
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler_ring: ring coupler function
        straight: straight function
        bend: 90 degrees bend function
        cross_section:
        **kwargs: cross_section settings


    .. code::

          bl-st-br
          |      |
          sl     sr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    gf.snap.assert_on_2nm_grid(gap)

    coupler_ring = gf.partial(
        coupler_ring,
        bend=bend,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section=cross_section,
        **kwargs
    )

    straight_side = gf.partial(
        straight, length=length_y, cross_section=cross_section, **kwargs
    )
    straight_top = gf.partial(
        straight, length=length_x, cross_section=cross_section, **kwargs
    )

    bend = gf.partial(bend, radius=radius, cross_section=cross_section, **kwargs)

    c = gf.Component()
    cb = c << coupler_ring()
    sl = c << straight_side()
    sr = c << straight_side()
    bl = c << bend()
    br = c << bend()
    st = c << straight_top()
    # st.mirror(p1=(0, 0), p2=(1, 0))

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

    c = ring_single(width=2, gap=1, layer=(2, 0), radius=7, length_y=1)
    print(c.ports)
    c.show(show_subports=False)

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show()
