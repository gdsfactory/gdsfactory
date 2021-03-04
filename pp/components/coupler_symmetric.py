from typing import Optional

import pp
from pp.component import Component
from pp.components.bend_s import bend_s
from pp.tech import TECH_SILICON_C, Tech
from pp.types import ComponentFactory


@pp.cell
def coupler_symmetric(
    bend: ComponentFactory = bend_s,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    tech: Optional[Tech] = None,
    wg_width: Optional[float] = None,
) -> Component:
    r"""Two coupled waveguides with bends.

    Args:
        bend: bend or factory
        gap:
        dy: port to port vertical spacing
        dx: bend length in x direction
        tech: Technology
        wg_width: waveguide width (defaults to tech.wg_width)

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler_symmetric()
      c.plot()

    .. code::

                    dx
                 |-----|
                  _____ E1
                 /         |
           _____/          |
      gap  _____           |  dy
                \          |
                 \_____    |
                        E0

    """
    tech = tech if isinstance(tech, Tech) else TECH_SILICON_C
    width = wg_width or tech.wg_width
    bend_component = (
        bend(width=width, height=(dy - gap - width) / 2, length=dx, tech=tech)
        if callable(bend)
        else bend
    )

    w = bend_component.ports["W0"].width
    y = (w + gap) / 2

    c = pp.Component()
    top_bend = bend_component.ref(position=(0, y), port_id="W0")
    bottom_bend = bend_component.ref(position=(0, -y), port_id="W0", v_mirror=True)

    c.add(top_bend)
    c.add(bottom_bend)

    c.absorb(top_bend)
    c.absorb(bottom_bend)

    c.add_port("W0", port=bottom_bend.ports["W0"])
    c.add_port("W1", port=top_bend.ports["W0"])

    c.add_port("E0", port=bottom_bend.ports["E0"])
    c.add_port("E1", port=top_bend.ports["E0"])
    c.length = bend_component.length
    c.min_bend_radius = bend_component.min_bend_radius
    return c


if __name__ == "__main__":
    c = coupler_symmetric(gap=0.2, wg_width=0.5, dx=5)
    c.show()
    c.pprint()

    for dyi in [2, 3, 4, 5]:
        c = coupler_symmetric(gap=0.2, wg_width=0.5, dy=dyi, dx=10.0)
        print(f"dy={dyi}, min_bend_radius = {c.min_bend_radius}")
