from __future__ import annotations

__all__ = ["ring_single_dut"]

from typing import Any

import gdsfactory as gf
from gdsfactory import component_functions as cf
from gdsfactory.component import Component
from gdsfactory.typings import ComponentSpec

from .._schematic import ring_single_schematic


@gf.cell_with_module_name(schematic_function=ring_single_schematic, tags=["rings"])
def ring_single_dut(
    component: ComponentSpec = "straight",
    gap: float = 0.2,
    length_x: float = 4,
    length_y: float = 0,
    radius: float | None = None,
    coupler: ComponentSpec = "coupler_ring",
    bend: ComponentSpec = "bend_euler",
    with_component: bool = True,
    port_name: str = "o1",
    length_extension: float | None = None,
    **kwargs: Any,
) -> Component:
    """Single bus ring made of two couplers (ct: top, cb: bottom) connected.

    with two vertical straights (wyl: left, wyr: right) (Component Under Test) in
    the middle to extract loss from quality factor.

    Args:
        component: device under test.
        gap: in um.
        length_x: in um.
        length_y: in um.
        radius: in um. Default is None, which uses the default radius of the cross_section.
        coupler: coupler function.
        bend: bend function.
        with_component: True adds component. False adds waveguide.
        port_name: for component input.
        length_extension: optional length extension for the coupler bottom ports.
        kwargs: cross_section settings.

    Args:
        with_component: if False changes component for just a straight.

          bl-wt-br
          |      | length_y
          wl     component
          |      |
         --==cb==-- gap

          length_x
    """
    return cf.ring_single_dut(
        component=component,
        gap=gap,
        length_x=length_x,
        length_y=length_y,
        radius=radius,
        coupler=coupler,
        bend=bend,
        with_component=with_component,
        port_name=port_name,
        length_extension=length_extension,
        **kwargs,
    )
