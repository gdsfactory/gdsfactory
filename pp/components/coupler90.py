from typing import Optional

from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.bend_euler import bend_euler
from pp.components.waveguide import waveguide
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


@cell
def coupler90(
    radius: float = 10.0,
    gap: float = 0.2,
    width: Optional[float] = None,
    waveguide_factory: ComponentFactory = waveguide,
    bend90_factory: ComponentFactory = bend_circular,
    tech: Tech = TECH_SILICON_C,
    **kwargs
) -> Component:
    r"""Waveguide coupled to a bend.

    Args:
        radius: um
        gap: um
        waveguide_factory: for Waveguide
        bend90_factory: for bend
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
    width = width or tech.wg_width

    c = Component()
    wg = c << waveguide_factory(length=radius, width=width, tech=tech)
    bend = c << bend90_factory(radius=radius, width=width, tech=tech, **kwargs)

    pbw = bend.ports["W0"]
    bend.movey(pbw.midpoint[1] + gap + width)

    # This component is a leaf cell => using absorb
    c.absorb(wg)
    c.absorb(bend)

    c.add_port("E0", port=wg.ports["E0"])
    c.add_port("N0", port=bend.ports["N0"])
    c.add_port("W0", port=wg.ports["W0"])
    c.add_port("W1", port=bend.ports["W0"])

    return c


def coupler90euler(
    radius: float = 10.0,
    gap: float = 0.2,
    waveguide_factory: ComponentFactory = waveguide,
    bend90_factory: ComponentFactory = bend_euler,
    **kwargs
):
    return coupler90(
        radius=radius,
        gap=gap,
        waveguide_factory=waveguide_factory,
        bend90_factory=bend90_factory,
        **kwargs
    )


if __name__ == "__main__":
    c = coupler90(gap=0.3)
    c << coupler90euler(gap=0.3, use_eff=True)
    c.show()
    c.pprint()
    # print(c.ports)
