import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.coupler_ring import coupler_ring
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.components.taper import taper
from gdsfactory.config import call_if_func
from gdsfactory.snap import assert_on_2nm_grid


@gf.cell
def ring_single_dut(
    component,
    wg_width=0.5,
    gap=0.2,
    length_x=4,
    radius=5,
    length_y=0,
    coupler=coupler_ring,
    straight=straight_function,
    bend=bend_euler,
    with_dut=True,
    **kwargs
):
    """Single bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical straights (wyl: left, wyr: right)
    DUT (Device Under Test) in the middle to extract loss from quality factor


    Args:
        with_dut: if False changes dut for just a straight

    .. code::

          bl-wt-br
          |      | length_y
          wl     dut
          |      |
         --==cb==-- gap

          length_x
    """
    dut = call_if_func(component)
    dut.auto_rename_ports()

    assert_on_2nm_grid(gap)

    coupler = call_if_func(coupler, gap=gap, radius=radius, length_x=length_x, **kwargs)
    straight_side = call_if_func(
        straight, width=wg_width, length=length_y + dut.xsize, **kwargs
    )
    straight_top = call_if_func(straight, width=wg_width, length=length_x, **kwargs)
    bend = call_if_func(bend, width=wg_width, radius=radius, **kwargs)

    c = Component()
    cb = c << coupler
    wl = c << straight_side
    if with_dut:
        d = c << dut
    else:
        d = c << straight_side
    bl = c << bend
    br = c << bend
    wt = c << straight_top

    wl.connect(port=2, destination=cb.ports[2])
    bl.connect(port=2, destination=wl.ports[1])

    wt.connect(port=1, destination=bl.ports[1])
    br.connect(port=2, destination=wt.ports[2])
    d.connect(port=1, destination=br.ports[1])
    c.add_port(2, port=cb.ports[2])
    c.add_port(1, port=cb.ports[1])
    return c


if __name__ == "__main__":
    c = ring_single_dut(component=taper(width2=3))
    c.show()
