from typing import Callable, Tuple

from pp.cell import cell
from pp.component import Component
from pp.components.waveguide import waveguide
from pp.layers import LAYER


@cell
def waveguide_pin(
    length: float = 10.0,
    width: float = 0.5,
    width_i: float = 0.0,
    width_p: float = 1.0,
    width_n: float = 1.0,
    width_pp: float = 1.0,
    width_np: float = 1.0,
    width_ppp: float = 1.0,
    width_npp: float = 1.0,
    layer_p: Tuple[int, int] = LAYER.P,
    layer_n: Tuple[int, int] = LAYER.N,
    layer_pp: Tuple[int, int] = LAYER.Pp,
    layer_np: Tuple[int, int] = LAYER.Np,
    layer_ppp: Tuple[int, int] = LAYER.Ppp,
    layer_npp: Tuple[int, int] = LAYER.Npp,
    waveguide_factory: Callable = waveguide,
) -> Component:
    """PN doped waveguide

        .. code::


                               |<------width------>|
                                ____________________
                               |     |       |     |
            ___________________|     |       |     |__________________________|
                                     |       |                                |
                P++     P+     P     |   I   |     N        N+         N++    |
            __________________________________________________________________|
                                                                              |
                                     |width_i| width_n | width_np | width_npp |
                                        0    oi        on        onp         onpp

    .. plot::
      :include-source:

      import pp

      c = pp.c.waveguide_pin(length=10)
      pp.plotgds(c)

    """
    c = Component()
    w = c << waveguide_factory(length=length, width=width)
    c.absorb(w)

    oi = width_i / 2
    on = oi + width_n
    onp = oi + width_n + width_np
    onpp = oi + width_n + width_np + width_npp

    # N doping
    c.add_polygon([(0, oi), (length, oi), (length, onpp), (0, onpp)], layer=layer_n)

    if layer_np:
        c.add_polygon(
            [(0, on), (length, on), (length, onpp), (0, onpp)], layer=layer_np
        )
    if layer_npp:
        c.add_polygon(
            [(0, onp), (length, onp), (length, onpp), (0, onpp)], layer=layer_npp
        )

    oi = -width_i / 2
    op = oi - width_p
    opp = oi - width_p - width_pp
    oppp = oi - width_p - width_pp - width_ppp

    # P doping
    c.add_polygon([(0, oi), (length, oi), (length, oppp), (0, oppp)], layer=layer_p)
    if layer_pp:
        c.add_polygon(
            [(0, op), (length, op), (length, oppp), (0, oppp)], layer=layer_pp
        )
    if layer_ppp:
        c.add_polygon(
            [(0, opp), (length, opp), (length, oppp), (0, oppp)], layer=layer_ppp
        )

    return c


if __name__ == "__main__":
    import pp

    c = waveguide_pin(width_i=1)
    pp.show(c)
