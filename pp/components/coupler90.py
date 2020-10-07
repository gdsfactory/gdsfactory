import pp
from pp.components.waveguide import waveguide
from pp.components.bend_circular import bend_circular
from pp.name import autoname
from pp.component import Component
from typing import Callable


@autoname
def coupler90(
    bend_radius: float = 10.0,
    width: float = 0.5,
    gap: float = 0.2,
    waveguide_factory: Callable = waveguide,
    bend90_factory: Callable = bend_circular,
) -> Component:
    """ Waveguide coupled to a bend with gap

    Args:
        bend_radius: um
        width: waveguide width (um)
        gap: um

    .. plot::
      :include-source:

      import pp
      c = pp.c.coupler90()
      pp.plotgds(c)

    """
    # pp.drc.assert_on_1nm_grid((width + gap) / 2)
    y = pp.drc.snap_to_1nm_grid((width + gap) / 2)

    c = Component()

    wg = c << waveguide_factory(length=bend_radius, width=width, pins=False)
    bend = c << bend90_factory(radius=bend_radius, width=width, pins=False)

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


if __name__ == "__main__":
    c = coupler90(width=0.45, gap=0.3)
    pp.show(c)
    # print(c.ports)
