import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.components.straight import straight as straight_function
from gdsfactory.types import ComponentSpec, CrossSectionSpec


@gf.cell
def coupler_asymmetric(
    bend: ComponentSpec = bend_s,
    straight: ComponentSpec = straight_function,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    cross_section: CrossSectionSpec = "strip",
    **kwargs
) -> Component:
    """Bend coupled to straight waveguide.

    Args:
        bend: spec.
        straight: straight spec.
        gap: um.
        dy: port to port vertical spacing.
        dx: bend length in x direction.
        cross_section: spec.
        kwargs: cross_section settings.

    .. code::

                        dx
                     |-----|
                      _____ o2
                     /         |
               _____/          |
         gap o1____________    |  dy
                            o3
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    width = x.width
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
    c.add_port(
        name="o1", center=(0, 0), width=port_width, orientation=180, cross_section=x
    )
    c.add_port(name="o3", port=bottom_bend.ports["o2"])
    c.add_port(name="o2", port=wg.ports["o2"])

    return c


if __name__ == "__main__":
    c = coupler_asymmetric(gap=0.4, layer=(2, 0))
    c.show(show_ports=True)
