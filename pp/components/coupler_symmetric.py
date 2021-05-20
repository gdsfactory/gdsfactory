import pp
from pp.component import Component
from pp.components.bend_s import bend_s
from pp.cross_section import cross_section, get_waveguide_settings
from pp.types import ComponentFactory


@pp.cell_with_validator
def coupler_symmetric(
    bend: ComponentFactory = bend_s,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    waveguide: str = "strip",
    **kwargs,
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend or factory
        gap:
        dy: port to port vertical spacing
        dx: bend length in x direction
        waveguide: from TECH.waveguide
        kwargs: waveguide_settings

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
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    x = cross_section(**waveguide_settings)
    width = x.info["width"]
    bend_component = (
        bend(
            height=(dy - gap - width) / 2,
            length=dx,
            **waveguide_settings,
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
    c = coupler_symmetric(gap=0.2, width=0.9, dx=5, waveguide="nitride")
    c.show()
    c.pprint()

    for dyi in [2, 3, 4, 5]:
        c = coupler_symmetric(gap=0.2, width=0.5, dy=dyi, dx=10.0, waveguide="nitride")
        print(f"dy={dyi}, min_bend_radius = {c.min_bend_radius}")
