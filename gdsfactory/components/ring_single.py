from typing import Optional

import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring as coupler_ring_function
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.config import call_if_func
from gdsfactory.snap import assert_on_2nm_grid
from gdsfactory.types import ComponentFactory


@gf.cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.010,
    coupler_ring: ComponentFactory = coupler_ring_function,
    straight: ComponentFactory = straight_function,
    bend: Optional[ComponentFactory] = None,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical straights (wl: left, wr: right)
    two bends (bl, br) and horizontal straight (wg: top)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical straight length
        coupler_ring: ring coupler function
        straight: straight function
        bend: 90 degrees bend function
        waveguide: settings for cross_section
        kwargs: overwrites waveguide_settings


    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    assert_on_2nm_grid(gap)

    coupler_ring_component = (
        coupler_ring(
            bend=bend,
            gap=gap,
            radius=radius,
            length_x=length_x,
            waveguide=waveguide,
            **kwargs
        )
        if callable(coupler_ring)
        else coupler_ring
    )
    straight_side = call_if_func(
        straight, length=length_y, waveguide=waveguide, **kwargs
    )
    straight_top = call_if_func(
        straight, length=length_x, waveguide=waveguide, **kwargs
    )

    bend = bend or bend_euler
    bend_ref = (
        bend(radius=radius, waveguide=waveguide, **kwargs) if callable(bend) else bend
    )

    c = Component()
    cb = c << coupler_ring_component
    wl = c << straight_side
    wr = c << straight_side
    bl = c << bend_ref
    br = c << bend_ref
    wt = c << straight_top
    # wt.mirror(p1=(0, 0), p2=(1, 0))

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="E0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["W0"])
    wr.connect(port="W0", destination=br.ports["W0"])
    wr.connect(port="E0", destination=cb.ports["N1"])  # just for netlist

    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    return c


if __name__ == "__main__":
    # c = ring_single(layer=(2, 0), cross_section_factory=gf.cross_section.pin, width=1)

    c = ring_single(width=2, gap=1)
    print(c.ports)
    c.show()

    # cc = gf.add_pins(c)
    # print(c.settings)
    # print(c.get_settings())
    # cc.show()
