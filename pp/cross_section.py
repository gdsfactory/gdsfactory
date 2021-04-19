"""You can define a path with a list of points combined with a cross-section.

The CrossSection object extrudes a path

Based on phidl.device_layout.CrossSection
"""

from typing import Iterable, Optional, Tuple

from phidl.device_layout import CrossSection

from pp.layers import LAYER

Layer = Tuple[int, int]


def strip(
    width: float = 0.5,
    layer: Layer = LAYER.WG,
    cladding_offset: float = 0,
    layers_cladding: Optional[Iterable[Layer]] = None,
) -> CrossSection:
    """Returns a fully etched straight CrossSection."""

    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=["in", "out"])

    layers_cladding = layers_cladding or []
    for layer_cladding in layers_cladding:
        x.add(width=width + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
    )
    x.info = s
    return x


def metal1(
    width: float = 2.0,
    layer: Layer = LAYER.M1,
    cladding_offset: float = 3,
    layers_cladding: Optional[Iterable[Layer]] = (LAYER.WGCLAD,),
) -> CrossSection:
    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=["in", "out"])

    layers_cladding = layers_cladding or []
    for layer_cladding in layers_cladding:
        x.add(width=width + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
    )
    x.info = s
    return x


def pin(
    width: float,
    layer: Layer = LAYER.WG,
    layer_slab: Layer = LAYER.SLAB90,
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
    cladding_offset: float = 0,
    layers_cladding: Optional[Iterable[Layer]] = None,
) -> CrossSection:

    """PIN doped straight.

    .. code::

                                   layer
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

    """
    x = CrossSection()
    x.add(width=width, offset=0, layer=layer, ports=["in", "out"])

    oi = width_i / 2
    on = oi + width_n
    onp = oi + width_n + width_np
    onpp = oi + width_n + width_np + width_npp

    op = -oi - width_p
    opp = op - width_pp
    oppp = opp - width_ppp

    offset_n = (oi + on) / 2
    offset_np = (on + onp) / 2
    offset_npp = (onp + onpp) / 2

    offset_p = (-oi + op) / 2
    offset_pp = (op + opp) / 2
    offset_ppp = (opp + oppp) / 2

    width_slab = abs(onpp) + abs(oppp)
    x.add(width=width_slab, offset=0, layer=layer_slab)

    x.add(width=width_n, offset=offset_n, layer=layer_n)
    x.add(width=width_np, offset=offset_np, layer=layer_np)
    x.add(width=width_npp, offset=offset_npp, layer=layer_npp)

    x.add(width=width_p, offset=offset_p, layer=layer_p)
    x.add(width=width_pp, offset=offset_pp, layer=layer_pp)
    x.add(width=width_ppp, offset=offset_ppp, layer=layer_ppp)

    for layer_cladding in layers_cladding or []:
        x.add(width=width_slab + 2 * cladding_offset, offset=0, layer=layer_cladding)

    s = dict(
        width=width,
        layer=layer,
        cladding_offset=cladding_offset,
        layers_cladding=layers_cladding,
    )
    x.info = s
    return x


if __name__ == "__main__":
    import pp

    P = pp.path.euler(radius=10, use_eff=True)
    # P = euler()
    # P = pp.Path()
    # P.append(pp.path.straight(length=5))
    # P.append(pp.path.arc(radius=10, angle=90))
    # P.append(pp.path.spiral())

    # Create a blank CrossSection
    # X = CrossSection()
    # X.add(width=2.0, offset=-4, layer=LAYER.HEATER, ports=["HW1", "HE1"])
    # X.add(width=0.5, offset=0, layer=LAYER.SLAB90, ports=["in", "out"])
    # X.add(width=2.0, offset=4, layer=LAYER.HEATER, ports=["HW0", "HE0"])

    # Combine the Path and the CrossSection into a Component
    X = pin(width=0.5, width_i=0.5)
    x = strip(width=0.5)
    c = pp.path.component(P, X)

    # c = pp.path.component(P, strip(width=2, layer=LAYER.WG, cladding_offset=3))

    # c = pp.add_pins(c)
    # c << pp.components.bend_euler(radius=10)
    # c << pp.components.bend_circular(radius=10)
    c.show()
