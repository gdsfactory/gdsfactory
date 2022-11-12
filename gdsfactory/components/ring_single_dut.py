import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.types import ComponentSpec

taper2 = gf.partial(taper, width2=3)


@gf.cell
def ring_single_dut(
    component=taper2,
    gap: float = 0.2,
    length_x: float = 4,
    length_y: float = 0,
    radius: float = 5.0,
    coupler: ComponentSpec = coupler_ring,
    straight: ComponentSpec = straight_function,
    bend: ComponentSpec = bend_euler,
    with_component: bool = True,
    port_name: str = "o1",
    **kwargs
) -> Component:
    """Single bus ring made of two couplers (ct: top, cb: bottom) connected.

    with two vertical straights (wyl: left, wyr: right) (Component Under Test) in
    the middle to extract loss from quality factor.

    Args:
        component: dut.
        gap:
        length:
        length_y:
        radius:
        coupler: coupler function
        straight: straight function
        bend: bend function
        with_component:
        port_name:
        kwargs: cross_section settings.

    Args:
        with_component: if False changes component for just a straight

    .. code::

          bl-wt-br
          |      | length_y
          wl     component
          |      |
         --==cb==-- gap

          length_x
    """
    component = gf.get_component(component)
    assert_on_2nm_grid(gap)

    coupler = gf.get_component(
        coupler, gap=gap, length_x=length_x, radius=radius, **kwargs
    )
    straight_side = gf.get_component(
        straight, length=length_y + component.xsize, **kwargs
    )
    straight_top = gf.get_component(straight, length=length_x, **kwargs)
    bend = gf.get_component(bend, radius=radius, **kwargs)

    c = Component()
    c.component = component
    cb = c << coupler
    wl = c << straight_side
    dut = c << component if with_component else c << straight_side
    bl = c << bend
    br = c << bend
    wt = c << straight_top

    wl.connect(port="o2", destination=cb.ports["o2"])
    bl.connect(port="o2", destination=wl.ports["o1"])

    wt.connect(port="o1", destination=bl.ports["o1"])
    br.connect(port="o2", destination=wt.ports["o2"])
    dut.connect(port=port_name, destination=br.ports["o1"])

    c.add_port("o2", port=cb.ports["o4"])
    c.add_port("o1", port=cb.ports["o1"])
    c.copy_child_info(component)
    return c


if __name__ == "__main__":
    c = ring_single_dut()
    c.show(show_ports=True)
