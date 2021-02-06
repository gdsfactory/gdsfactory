import pp
from pp.cell import cell
from pp.component import Component
from pp.components.bend_circular import bend_circular
from pp.components.bend_euler import bend_euler
from pp.components.waveguide import waveguide
from pp.types import ComponentFactory


@cell
def coupler90(
    bend_radius: float = 10.0,
    width: float = 0.5,
    gap: float = 0.2,
    waveguide_factory: ComponentFactory = waveguide,
    bend90_factory: ComponentFactory = bend_circular,
) -> Component:
    r"""Waveguide coupled to a bend.

    Args:
        bend_radius: um
        width: waveguide width (um)
        gap: um

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
    y = pp.snap_to_grid((width + gap) / 2)

    c = Component()
    wg = c << waveguide_factory(length=bend_radius, width=width,)
    bend = c << bend90_factory(radius=bend_radius, width=width)

    pbw = bend.ports["W0"]
    bend.movey(pbw.midpoint[1] + gap + width)

    # This component is a leaf cell => using absorb
    c.absorb(wg)
    c.absorb(bend)

    port_width = 2 * width + gap
    c.add_port(port=wg.ports["E0"], name="E0")
    c.add_port(port=bend.ports["N0"], name="N0")
    c.add_port(name="W0", midpoint=[0, y], width=port_width, orientation=180)
    return c


def coupler90euler(
    bend_radius: float = 10.0,
    width: float = 0.5,
    gap: float = 0.2,
    waveguide_factory: ComponentFactory = waveguide,
    bend90_factory: ComponentFactory = bend_euler,
):
    return coupler90(
        bend_radius=bend_radius,
        width=width,
        gap=gap,
        waveguide_factory=waveguide_factory,
        bend90_factory=bend90_factory,
    )


if __name__ == "__main__":
    c = coupler90(width=0.45, gap=0.3)
    c << coupler90euler(width=0.45, gap=0.3)
    c.show()
    # print(c.ports)
