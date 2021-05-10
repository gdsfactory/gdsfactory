from pydantic import validate_arguments

from pp.cell import cell
from pp.component import Component
from pp.components.bend_s import bend_s
from pp.cross_section import cross_section
from pp.cross_section import get_cross_section_settings
from pp.types import ComponentFactory


@cell
@validate_arguments
def coupler_symmetric(
    bend: ComponentFactory = bend_s,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section_name: str = "strip",
    **kwargs,
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend or factory
        gap:
        dy: port to port vertical spacing
        dx: bend length in x direction
        cross_section_name: from TECH.waveguide
        kwargs: cross_section_settings

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
    cross_section_settings = get_cross_section_settings(cross_section_name, **kwargs)
    x = cross_section(**cross_section_settings)
    width = x.info["width"]
    bend_component = (
        bend(
            height=(dy - gap - width) / 2,
            length=dx,
            **cross_section_settings,
        )
        if callable(bend)
        else bend
    )

    w = bend_component.ports["W0"].width
    y = (w + gap) / 2

    c = Component()
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
    c = coupler_symmetric(gap=0.2, width=0.9, dx=5, cross_section_name="nitride")
    c.show()
    c.pprint()

    for dyi in [2, 3, 4, 5]:
        c = coupler_symmetric(
            gap=0.2, width=0.5, dy=dyi, dx=10.0, cross_section_name="nitride"
        )
        print(f"dy={dyi}, min_bend_radius = {c.min_bend_radius}")
