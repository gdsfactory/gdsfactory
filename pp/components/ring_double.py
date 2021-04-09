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
    radius: float = 10.0,
    length_x: float = 0.01,
    length_y: float = 0.01,
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
        cross_section_factory_inner: for inner bend
        cross_section_factory_outer: for outer waveguide
        tech: Technology with default values

    .. code::

         --==ct==--
          |      |
          wl     wr length_y
          |      |
         --==cb==-- gap

          length_x

    """
    assert_on_2nm_grid(gap)

    coupler_component = (
        coupler(
            gap=gap,
            radius=radius,
            length_x=length_x,
            width=width,
            layer=layer,
            cross_section_factory_inner=cross_section_factory_inner,
            cross_section_factory_outer=cross_section_factory_outer,
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
        cross_section_factory=cross_section_factory_inner,
        tech=tech,
    )

    c = Component()
    cb = c.add_ref(coupler_component)
    ct = c.add_ref(coupler_component)
    wl = c.add_ref(waveguide)
    wr = c.add_ref(waveguide)

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
