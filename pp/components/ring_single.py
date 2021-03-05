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
    cross_section_factory: Optional[CrossSectionFactory] = None,
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
        bend: bend function
        pins: add pins
        tech: Technology with default values


    .. code::

          bl-wt-br
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring_single(gap=0.2, length_x=4, length_y=0.1, radius=5)
      c.plot()

    """
    assert_on_2nm_grid(gap)

    coupler_ring = (
        coupler(
            bend=bend,
            gap=gap,
            radius=radius,
            length_x=length_x,
            width=width,
            layer=layer,
            cross_section_factory=cross_section_factory,
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
        cross_section_factory=cross_section_factory,
        tech=tech,
    )
    waveguide_top = call_if_func(
        waveguide,
        length=length_x,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory,
        tech=tech,
    )

    bend = bend or bend_euler
    bend_ref = (
        bend(
            radius=radius,
            width=width,
            layer=layer,
            cross_section_factory=cross_section_factory,
            tech=tech,
        )
        if callable(bend)
        else bend
    )

    c = Component()
    cb = c << coupler_ring
    wl = c << waveguide_side
    wr = c << waveguide_side
    bl = c << bend_ref
    br = c << bend_ref
    wt = c << waveguide_top

    wl.connect(port="E0", destination=cb.ports["N0"])
    bl.connect(port="N0", destination=wl.ports["W0"])

    wt.connect(port="W0", destination=bl.ports["W0"])
    br.connect(port="N0", destination=wt.ports["E0"])
    wr.connect(port="W0", destination=br.ports["W0"])
    wr.connect(port="E0", destination=cb.ports["N1"])  # just for netlist

    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":

    c = ring_single(layer=(2, 0))
    print(c.ports)
    c.show()
    # cc = pp.add_pins(c)
    # print(c.settings)
    # print(c.get_settings())
    # cc.show()
