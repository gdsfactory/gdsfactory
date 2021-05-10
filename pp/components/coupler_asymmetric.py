from pydantic import validate_arguments

from pp.cell import cell
from pp.component import Component
from pp.components.bend_s import bend_s
from pp.components.straight import straight as straight_function
from pp.cross_section import cross_section
from pp.cross_section import get_cross_section_settings
from pp.types import ComponentFactory


@cell
@validate_arguments
def coupler_asymmetric(
    bend: ComponentFactory = bend_s,
    straight: ComponentFactory = straight_function,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section_name: str = "strip",
    **kwargs
) -> Component:
    """bend coupled to straight waveguide

    Args:
        bend:
        straight: straight factory
        gap: um
        dy: port to port vertical spacing
        dx: bend length in x direction
        cross_section_factory: function that returns a cross_section
        **cross_section_settings

    .. code::
                        dx
                     |-----|
                      _____ E1
                     /         |
               _____/          |
          gap  ____________    |  dy
                            E0

    """
    cross_section_settings = get_cross_section_settings(cross_section_name, **kwargs)
    x = cross_section(**cross_section_settings)
    width = x.info["width"]
    bend_component = (
        bend(height=(dy - gap - width), length=dx, **cross_section_settings)
        if callable(bend)
        else bend
    )
    wg = straight(**cross_section_settings) if callable(straight) else straight

    w = bend_component.ports["W0"].width
    y = (w + gap) / 2

    c = Component()
    wg = wg.ref(position=(0, y), port_id="W0")
    bottom_bend = bend_component.ref(position=(0, -y), port_id="W0", v_mirror=True)

    c.add(wg)
    c.add(bottom_bend)

    # Using absorb here to have a flat cell and avoid
    # to have deeper hierarchy than needed
    c.absorb(wg)
    c.absorb(bottom_bend)

    port_width = 2 * w + gap
    c.add_port(name="W0", midpoint=[0, 0], width=port_width, orientation=180)
    c.add_port(port=bottom_bend.ports["E0"], name="E0")
    c.add_port(port=wg.ports["E0"], name="E1")

    return c


if __name__ == "__main__":
    c = coupler_asymmetric(gap=0.4, cross_section_name="nitride")
    c.show()
