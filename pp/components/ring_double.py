from typing import Optional

import pp
from pp.cell import cell
from pp.component import Component
from pp.components.coupler_ring import coupler_ring
from pp.components.waveguide import waveguide as waveguide_function
from pp.config import call_if_func
from pp.cross_section import CrossSectionFactory
from pp.snap import assert_on_2nm_grid
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Layer


@cell
def ring_double(
    gap: float = 0.2,
    length_x: float = 0.01,
    radius: float = 10.0,
    length_y: float = 0.01,
    coupler: ComponentFactory = coupler_ring,
    waveguide: ComponentFactory = waveguide_function,
    bend: Optional[ComponentFactory] = None,
    pins: bool = False,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
) -> Component:
    """Double bus ring made of two couplers (ct: top, cb: bottom)
    connected with two vertical waveguides (wyl: left, wyr: right)

    Args:
        gap: gap between for coupler
        radius: for the bend and coupler
        length_x: ring coupler length
        length_y: vertical waveguide length
        coupler: ring coupler function
        waveguide: waveguide function
        bend: bend function
        pins: add pins
        width: waveguide width
        layer:
        cross_section_factory: to extrude the paths
        tech: Technology with default values

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    .. plot::
      :include-source:

      import pp

      c = pp.c.ring_double(gap=0.2, length_x=4, length_y=0.1, radius=5)
      c.plot()
    """
    assert_on_2nm_grid(gap)

    coupler = (
        coupler(
            gap=gap,
            radius=radius,
            length_x=length_x,
            width=width,
            layer=layer,
            cross_section_factory=cross_section_factory,
            bend=bend,
            tech=tech,
        )
        if callable(coupler)
        else coupler
    )
    waveguide = call_if_func(
        waveguide,
        length=length_y,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory,
        tech=tech,
    )

    c = Component()
    cb = c << coupler
    ct = c << coupler
    wl = c << waveguide
    wr = c << waveguide

    wl.connect(port="E0", destination=cb.ports["N0"])
    ct.connect(port="N1", destination=wl.ports["W0"])
    wr.connect(port="W0", destination=ct.ports["N0"])
    cb.connect(port="N1", destination=wr.ports["E0"])
    c.add_port("E0", port=cb.ports["E0"])
    c.add_port("W0", port=cb.ports["W0"])
    c.add_port("E1", port=ct.ports["W0"])
    c.add_port("W1", port=ct.ports["E0"])
    if pins:
        pp.add_pins_to_references(c)
    return c


if __name__ == "__main__":

    c = ring_double()
    c.show()
