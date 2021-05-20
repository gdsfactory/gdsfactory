import pp
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.bend_euler import bend_euler
from pp.components.straight import straight as straight_function
from pp.cross_section import cross_section, get_waveguide_settings
from pp.types import ComponentFactory, ComponentOrFactory


@pp.cell_with_validator
def coupler90(
    gap: float = 0.2,
    radius: float = 10.0,
    straight: ComponentOrFactory = straight_function,
    bend: ComponentFactory = bend_euler,
    waveguide: str = "strip",
    **kwargs
) -> Component:
    r"""straight coupled to a bend.

    Args:
        gap: um
        radius: um
        straight: for straight
        bend: for bend
        kwargs:  waveguide_settings

    .. code::

             N0
             |
            /
           /
       W1_/
       W0____E0

    """
    c = Component()
    waveguide_settings = get_waveguide_settings(waveguide, **kwargs)
    waveguide_settings.update(radius=radius)
    x = cross_section(**waveguide_settings)

    bend90 = bend(**waveguide_settings) if callable(bend) else bend
    bend_ref = c << bend90
    straight_component = (
        straight(
            length=bend90.ports["N0"].midpoint[0] - bend90.ports["W0"].midpoint[0],
            **waveguide_settings
        )
        if callable(straight)
        else straight
    )

    wg_ref = c << straight_component

    width = x.info["width"]

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


def coupler90circular(bend: ComponentFactory = bend_circular, **kwargs):
    return coupler90(bend=bend, **kwargs)


if __name__ == "__main__":
    # c = coupler90circular(gap=0.3)
    # c << coupler90(gap=0.3)
    c = coupler90(radius=3, width=1)
    c.show()
    c.pprint()
    # print(c.ports)
