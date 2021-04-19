from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.coupler_ring import coupler_ring
from pp.components.straight import straight as straight_function
from pp.components.taper import taper
from pp.config import call_if_func
from pp.port import rename_ports_by_orientation
from pp.snap import assert_on_2nm_grid
from pp.types import CrossSectionFactory


@cell
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
    cross_section_factory: Optional[CrossSectionFactory] = None,
    **cross_section_settings
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
    dut = rename_ports_by_orientation(dut)

    assert_on_2nm_grid(gap)

    coupler = call_if_func(
        coupler,
        gap=gap,
        radius=radius,
        length_x=length_x,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )
    straight_side = call_if_func(
        straight,
        width=wg_width,
        length=length_y + dut.xsize,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )
    straight_top = call_if_func(
        straight,
        width=wg_width,
        length=length_x,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )
    bend = call_if_func(
        bend,
        width=wg_width,
        radius=radius,
        cross_section_factory=cross_section_factory,
        **cross_section_settings
    )

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

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="W0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["E0"])
    d.connect(port="W0", destination=br.ports["W0"])
    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    return c


if __name__ == "__main__":
    c = ring_single_dut(component=taper(width2=3))
    c.show()
