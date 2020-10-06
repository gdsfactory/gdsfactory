from typing import Callable, List, Tuple
from pp.component import Component
from pp.name import autoname
from pp.layers import LAYER
from pp.drc import assert_on_1nm_grid
from pp.components.coupler_symmetric import coupler_symmetric
from pp.components.coupler_straight import coupler_straight


@autoname
def coupler(
    wg_width: float = 0.5,
    gap: float = 0.236,
    length: float = 20.007,
    coupler_symmetric_factory: Callable = coupler_symmetric,
    coupler_straight: Callable = coupler_straight,
    layer: Tuple[int, int] = LAYER.WG,
    layers_cladding: List[Tuple[int, int]] = [LAYER.WGCLAD],
    cladding_offset: int = 3,
) -> Component:
    r""" symmetric coupler

    Args:
        gap
        length
        coupler_symmetric_factory
        coupler_straight

    .. code::

       W1 __                           __ E1
            \                         /
             \        length         /
              ======================= gap
             /                        \
           _/                          \_
        W0                              E0

            coupler_straight  coupler_symmetric_factory

    .. plot::
      :include-source:

      import pp

      c = pp.c.coupler(gap=0.2, length=10)
      pp.plotgds(c)

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
    )

    sr = c << sbend
    sl = c << sbend
    cs = c << coupler_straight(
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
    return c


if __name__ == "__main__":
    import pp

    c = coupler(length=10, pins=True)
    print(c.settings_changed)
    pp.show(c)
