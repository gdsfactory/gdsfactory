import gdsfactory as gf
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.contact import contact_heater_m3
from gdsfactory.components.coupler_ring import coupler_ring as _coupler_ring
from gdsfactory.components.straight import straight as _straight
from gdsfactory.types import ComponentFactory, CrossSectionFactory, Float2

contact_heater_m3_mini = gf.partial(contact_heater_m3, size=(4, 4))


@gf.cell
def ring_single_heater(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.6,
    coupler_ring: ComponentFactory = _coupler_ring,
    straight: ComponentFactory = _straight,
    bend: ComponentFactory = bend_euler,
    cross_section_heater: CrossSectionFactory = gf.cross_section.strip_heater_metal,
    cross_section: CrossSectionFactory = gf.cross_section.strip,
    contact: ComponentFactory = contact_heater_m3_mini,
    port_orientation: int = 90,
    contact_offset: Float2 = (0, 0),
    **kwargs
) -> gf.Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical straights (sl: left, sr: right)
    two bends (bl, br) and horizontal straight (wg: top)
    includes heater

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler_ring: ring coupler function
        straight: straight function
        bend: 90 degrees bend function
        cross_section_heater:
        cross_section:
        contact:
        port_orientation: for electrical ports to promote from contact
        kwargs: cross_section settings


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
        bend_cross_section=cross_section_heater,
        **kwargs
    )

    straight_side = gf.partial(
        straight, length=length_y, cross_section=cross_section_heater, **kwargs
    )
    straight_top = gf.partial(
        straight, length=length_x, cross_section=cross_section_heater, **kwargs
    )

    bend = gf.partial(bend, radius=radius, cross_section=cross_section_heater, **kwargs)

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

    c1 = c << contact()
    c2 = c << contact()
    c1.xmax = -length_x / 2 + cb.x - contact_offset[0]
    c2.xmin = +length_x / 2 + cb.x + contact_offset[0]
    c1.movey(contact_offset[1])
    c2.movey(contact_offset[1])
    c.add_ports(c1.get_ports_list(orientation=port_orientation), prefix="e1")
    c.add_ports(c2.get_ports_list(orientation=port_orientation), prefix="e2")
    c.auto_rename_ports()
    return c


if __name__ == "__main__":
    c = ring_single_heater(width=0.5, gap=1, layer=(2, 0), radius=10, length_y=1)
    print(c.ports)
    c.show(show_subports=False)

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.settings)
    # cc.show()
