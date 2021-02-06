from typing import Iterable, Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.coupler_straight import coupler_straight
from pp.components.coupler_symmetric import coupler_symmetric
from pp.config import conf
from pp.layers import LAYER
from pp.snap import assert_on_1nm_grid
from pp.types import ComponentFactory


@cell
def coupler(
    wg_width: float = 0.5,
    gap: float = 0.236,
    length: float = 20.007,
    coupler_symmetric_factory: ComponentFactory = coupler_symmetric,
    coupler_straight_factory: ComponentFactory = coupler_straight,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: Iterable[Tuple[int, int]] = (LAYER.WGCLAD,),
    cladding_offset: float = conf.tech.cladding_offset,
    dy: float = 5.0,
    dx: float = 10.0,
) -> Component:
    r"""symmetric coupler

    Args:
        wg_width: width of the waveguide
        gap
        length
        coupler_symmetric_factory
        coupler_straight_factory
        layer:
        layers_cladding: list of cladding layers
        cladding_offset: offset from waveguide to cladding edge
        dy: port to port vertical spacing
        dx: length of bend in x direction

    .. code::

               dx                                 dx
            |------|                           |------|
         W1 ________                           _______E1
                    \                         /           |
                     \        length         /            |
                      ======================= gap         | dy
                     /                       \            |
            ________/                         \_______    |
         W0                                           E0

              coupler_straight_factory  coupler_symmetric_factory

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler(gap=0.2, length=10)
      c.plot()

    """
    assert_on_1nm_grid(length)
    assert_on_1nm_grid(gap)
    c = Component()

    sbend = coupler_symmetric_factory(
        gap=gap,
        wg_width=wg_width,
        layer=layer,
        layers_cladding=layers_cladding,
        cladding_offset=cladding_offset,
        dy=dy,
        dx=dx,
    )

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight_factory(
        length=length,
        gap=gap,
        width=wg_width,
        layer=layer,
        layers_cladding=layers_cladding,
        cladding_offset=cladding_offset,
    )
    sl.connect("W0", destination=cs.ports["W0"])
    sr.connect("W0", destination=cs.ports["E0"])

    c.add_port("W1", port=sl.ports["E0"])
    c.add_port("W0", port=sl.ports["E1"])
    c.add_port("E0", port=sr.ports["E0"])
    c.add_port("E1", port=sr.ports["E1"])

    c.absorb(sl)
    c.absorb(sr)
    c.absorb(cs)
    c.length = sbend.length
    c.min_bend_radius = sbend.min_bend_radius
    return c


if __name__ == "__main__":

    # c = pp.Component()
    # cp1 = c << coupler(gap=0.2)
    # cp2 = c << coupler(gap=0.5)
    # cp1.ymin = 0
    # cp2.ymin = 0

    c = coupler(length=1, dy=2, gap=0.2)
    c = coupler(gap=0.2)
    # print(c.settings_changed)
    c.show()
