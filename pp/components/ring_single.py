from typing import Optional

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_euler import bend_euler
from pp.components.coupler_ring import coupler_ring
from pp.components.waveguide import waveguide as waveguide_function
from pp.config import call_if_func
from pp.cross_section import CrossSectionFactory
from pp.snap import assert_on_2nm_grid
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Layer


@cell
def ring_single(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4.0,
    length_y: float = 0.010,
    coupler: ComponentFactory = coupler_ring,
    waveguide: ComponentFactory = waveguide_function,
    bend: Optional[ComponentFactory] = None,
    pins: bool = False,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory_inner: Optional[CrossSectionFactory] = None,
    cross_section_factory_outer: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
) -> Component:
    """Single bus ring made of a ring coupler (cb: bottom)
    connected with two vertical waveguides (wl: left, wr: right)
    two bends (bl, br) and horizontal waveguide (wg: top)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical waveguide length
        coupler: ring coupler function
        waveguide: waveguide function
        bend: 90 degrees bend function
        pins: add pins
        width: waveguide width
        layer:
        cross_section_factory_inner: for inner bend
        cross_section_factory_outer: for outer waveguide
        tech: Technology with default values


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
        coupler(
            bend=bend,
            gap=gap,
            radius=radius,
            length_x=length_x,
            width=width,
            layer=layer,
            cross_section_factory_inner=cross_section_factory_inner,
            cross_section_factory_outer=cross_section_factory_outer,
            tech=tech,
        )
        if callable(coupler)
        else coupler
    )
    waveguide_side = call_if_func(
        waveguide,
        length=length_y,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory_inner,
        tech=tech,
    )
    waveguide_top = call_if_func(
        waveguide,
        length=length_x,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory_inner,
        tech=tech,
    )

    bend = bend or bend_euler
    bend_ref = (
        bend(
            radius=radius,
            width=width,
            layer=layer,
            cross_section_factory=cross_section_factory_inner,
            tech=tech,
        )
        if callable(bend)
        else bend
    )

    c = Component()
    cb = c << coupler_ring_component
    wl = c << waveguide_side
    wr = c << waveguide_side
    bl = c << bend_ref
    br = c << bend_ref
    wt = c << waveguide_top
    # wt.mirror(p1=(0, 0), p2=(1, 0))

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="E0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["W0"])
    wr.connect(port="W0", destination=br.ports["W0"])
    wr.connect(port="E0", destination=cb.ports["N1"])  # just for netlist

    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":

    # c = ring_single(layer=(2, 0), radius=10, cross_section_factory_inner=pp.cross_section.pin)
    c = ring_single(layer=(2, 0), cross_section_factory_inner=pp.cross_section.pin)
    print(c.ports)
    c.show()
    # cc = pp.add_pins(c)
    # print(c.settings)
    # print(c.get_settings())
    # cc.show()
