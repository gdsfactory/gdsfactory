from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.bend_euler import bend_euler
from pp.components.waveguide import waveguide
from pp.cross_section import CrossSectionFactory
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory, Layer


@cell
def coupler90(
    radius: float = 10.0,
    gap: float = 0.2,
    waveguide_factory: ComponentFactory = waveguide,
    bend: ComponentFactory = bend_euler,
    width: float = TECH_SILICON_C.wg_width,
    layer: Layer = TECH_SILICON_C.layer_wg,
    cross_section_factory: Optional[CrossSectionFactory] = None,
    tech: Optional[Tech] = None,
    **kwargs
) -> Component:
    r"""Waveguide coupled to a bend.

    Args:
        radius: um
        gap: um
        waveguide_factory: for Waveguide
        bend: for bend
        tech: Technology

    .. code::

             N0
             |
            /
           /
       W0 =--- E0


    .. plot::
      :include-source:

      import pp
      c = pp.c.coupler90()
      c.plot()

    """
    c = Component()
    wg_ref = c << waveguide_factory(
        length=radius,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory,
        tech=tech,
    )
    bend_ref = c << bend(
        radius=radius,
        width=width,
        layer=layer,
        cross_section_factory=cross_section_factory,
        tech=tech,
        **kwargs
    )

    pbw = bend_ref.ports["W0"]
    bend_ref.movey(pbw.midpoint[1] + gap + width)

    # This component is a leaf cell => using absorb
    c.absorb(wg_ref)
    c.absorb(bend_ref)

    c.add_port("E0", port=wg_ref.ports["E0"])
    c.add_port("N0", port=bend_ref.ports["N0"])
    c.add_port("W0", port=wg_ref.ports["W0"])
    c.add_port("W1", port=bend_ref.ports["W0"])

    return c


def coupler90circular(
    radius: float = 10.0,
    gap: float = 0.2,
    waveguide_factory: ComponentFactory = waveguide,
    bend: ComponentFactory = bend_circular,
    **kwargs
):
    return coupler90(
        radius=radius, gap=gap, waveguide_factory=waveguide_factory, bend=bend, **kwargs
    )


if __name__ == "__main__":
    c = coupler90circular(gap=0.3)
    c << coupler90(gap=0.3)
    c.show()
    c.pprint()
    # print(c.ports)
