from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.straight import straight
from gdsfactory.components.taper import taper
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.typings import ComponentSpec

taper2 = partial(taper, width2=3)


@gf.cell
def ring_single_dut(
    component: ComponentSpec = taper2,
    gap: float = 0.2,
    length_x: float = 4,
    length_y: float = 0,
    radius: float = 5.0,
    coupler: ComponentSpec = coupler_ring,
    bend: ComponentSpec = bend_euler,
    with_component: bool = True,
    port_name: str = "o1",
    **kwargs,
) -> Component:
    """Single bus ring made of two couplers (ct: top, cb: bottom) connected.

    with two vertical straights (wyl: left, wyr: right) (Component Under Test) in
    the middle to extract loss from quality factor.

    Args:
        component: device under test.
        gap: in um.
        length: in um.
        length_y: in um.
        radius: in um.
        coupler: coupler function.
        bend: bend function.
        with_component: True adds component. False adds waveguide.
        port_name: for component input.
        kwargs: cross_section settings.

    Args:
        with_component: if False changes component for just a straight.

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

    component_xsize = component.get_ports_xsize()
    straight_side = straight(length=length_y + component_xsize, **kwargs)
    straight_top = straight(length=length_x, **kwargs)
    bend = gf.get_component(bend, radius=radius, **kwargs)

    c = Component()
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
    c.info["component"] = component.info
    return c


if __name__ == "__main__":
    c = ring_single_dut()
    c.show(show_ports=True)
