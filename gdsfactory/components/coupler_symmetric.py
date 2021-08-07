import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_s import bend_s
from gdsfactory.cross_section import StrOrDict, get_cross_section
from gdsfactory.types import ComponentFactory


@gf.cell
def coupler_symmetric(
    bend: ComponentFactory = bend_s,
    gap: float = 0.234,
    dy: float = 5.0,
    dx: float = 10.0,
    waveguide: StrOrDict = "strip",
    **kwargs,
) -> Component:
    r"""Two coupled straights with bends.

    Args:
        bend: bend or library
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
    x = get_cross_section(waveguide, **kwargs)
    width = x.info["width"]
    bend_component = (
        bend(
            height=(dy - gap - width) / 2,
            length=dx,
            waveguide=waveguide,
            **kwargs,
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
