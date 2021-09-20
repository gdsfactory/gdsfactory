import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.cross_section import strip
from gdsfactory.types import ComponentFactory, CrossSectionFactory


@gf.cell
def coupler_asymmetric(
    bend: ComponentFactory = bend_s,
    straight: ComponentFactory = straight_function,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """bend coupled to straight waveguide

    Args:
        bend:
        straight: straight library
        gap: um
        dy: port to port vertical spacing
        dx: bend length in x direction
        cross_section:
        **kwargs: cross_section settings

    .. code::

                        dx
                     |-----|
                      _____ o2
                     /         |
               _____/          |
         gap o1____________    |  dy
                            o3

    """
    x = cross_section(**kwargs)
    width = x.info["width"]
    bend_component = (
        bend(size=(dx, dy - gap - width), cross_section=cross_section, **kwargs)
        if callable(bend)
        else bend
    )
    wg = (
        straight(cross_section=cross_section, **kwargs)
        if callable(straight)
        else straight
    )

    w = bend_component.ports["o1"].width
    y = (w + gap) / 2

    c = Component()
    wg = wg.ref(position=(0, y), port_id="o1")
    bottom_bend = bend_component.ref(position=(0, -y), port_id="o1", v_mirror=True)

    c.add(wg)
    c.add(bottom_bend)

    # Using absorb here to have a flat cell and avoid
    # to have deeper hierarchy than needed
    c.absorb(wg)
    c.absorb(bottom_bend)

    port_width = 2 * w + gap
    c.add_port(name="o1", midpoint=[0, 0], width=port_width, orientation=180)
    c.add_port(port=bottom_bend.ports["o2"], name="o3")
    c.add_port(port=wg.ports["o2"], name="o2")

    return c


if __name__ == "__main__":
    c = coupler_asymmetric(gap=0.4, layer=(2, 0))
    c.show()
