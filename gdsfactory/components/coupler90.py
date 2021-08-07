import gdsfactory as gf
from gdsfactory.component import Component
from gdsfactory.components.bend_circular import bend_circular
from gdsfactory.components.bend_euler import bend_euler
from gdsfactory.components.straight import straight as straight
from gdsfactory.cross_section import StrOrDict, get_cross_section
from gdsfactory.types import ComponentFactory


@gf.cell
def coupler90(
    gap: float = 0.2,
    radius: float = 10.0,
    bend: ComponentFactory = bend_euler,
    waveguide: StrOrDict = "strip",
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
    kwargs.update(radius=radius)
    x = get_cross_section(waveguide, **kwargs)

    bend90 = bend(waveguide=waveguide, **kwargs) if callable(bend) else bend
    bend_ref = c << bend90
    straight_component = (
        straight(
            waveguide=waveguide,
            length=bend90.ports["N0"].midpoint[0] - bend90.ports["W0"].midpoint[0],
            **kwargs
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
