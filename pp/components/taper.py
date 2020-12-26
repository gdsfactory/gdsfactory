from typing import Any, List, Optional, Tuple

import pp
from pp.component import Component
from pp.config import TAPER_LENGTH


@pp.cell
def taper(
    length: float = TAPER_LENGTH,
    width1: float = 0.5,
    width2: Optional[float] = None,
    port: None = None,
    layer: Tuple[int, int] = pp.LAYER.WG,
    layers_cladding: List[Any] = None,
    cladding_offset: float = 3.0,
) -> Component:
    """ Linear taper.

    Args:
        length:
        width1:
        width2:
        port: can taper from a port instead of defining width1
        layer:
        layers_cladding:
        cladding_offset:

    .. plot::
      :include-source:

      import pp

      c = pp.c.taper(width1=0.5, width2=5, length=3)
      pp.plotgds(c)

    """
    layers_cladding = layers_cladding or []
    if isinstance(port, pp.Port) and width1 is None:
        width1 = port.width
    if width2 is None:
        width2 = width1

    y1 = width1 / 2
    y2 = width2 / 2

    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]

    c = pp.Component()
    c.add_polygon((xpts, ypts), layer=layer)
    c.add_port(name="1", midpoint=[0, 0], width=width1, orientation=180, layer=layer)
    c.add_port(name="2", midpoint=[length, 0], width=width2, orientation=0, layer=layer)

    o = cladding_offset
    ypts = [y1 + o, y2 + o, -y2 - o, -y1 - o]

    for layer in layers_cladding:
        c.add_polygon((xpts, ypts), layer=layer)

    c.info["length"] = length
    c.info["width1"] = width1
    c.info["width2"] = width2
    return c


@pp.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.15,
    w_slab2: float = 5.0,
) -> Component:
    """ taper strip to rib

    Args:
        length:
        width1:
        width2:
        w_slab1
        w_slab2

    .. plot::
      :include-source:

      import pp

      c = pp.c.taper_strip_to_ridge()
      pp.plotgds(c)

    """

    _taper_wg = taper(length=length, width1=width1, width2=width2, layer=pp.LAYER.WG)
    _taper_slab = taper(
        length=length, width1=w_slab1, width2=w_slab2, layer=pp.LAYER.SLAB90
    )

    c = pp.Component()
    for _t in [_taper_wg, _taper_slab]:
        taper_ref = _t.ref()
        c.add(taper_ref)
        c.absorb(taper_ref)

    c.info["length"] = length
    c.add_port(name="1", port=_taper_wg.ports["1"])
    c.add_port(name="wg_2", port=_taper_wg.ports["2"])
    c.add_port(name="slab_2", port=_taper_slab.ports["2"])

    return c


@pp.cell
def taper_strip_to_ridge_trenches(
    length=10.0,
    width=0.5,
    slab_offset=3.0,
    trench_width=2.0,
    trench_layer=pp.LAYER.SLAB90,
    wg_layer=pp.LAYER.WG,
    trench_offset_after_wg=0.1,
):

    c = pp.Component()
    width = pp.bias.width(width)
    y0 = width / 2 + trench_width - trench_offset_after_wg
    yL = width / 2 + trench_width - trench_offset_after_wg + slab_offset

    # waveguide
    x = [0, length, length, 0]
    yw = [y0, yL, -yL, -y0]
    c.add_polygon((x, yw), layer=wg_layer)

    # top trench
    ymin0 = width / 2
    yminL = width / 2
    ymax0 = width / 2 + trench_width
    ymaxL = width / 2 + trench_width + slab_offset
    x = [0, length, length, 0]
    ytt = [ymin0, yminL, ymaxL, ymax0]
    ytb = [-ymin0, -yminL, -ymaxL, -ymax0]
    c.add_polygon((x, ytt), layer=trench_layer)
    c.add_polygon((x, ytb), layer=trench_layer)

    c.add_port(name="W0", midpoint=[0, 0], width=width, orientation=180, layer=wg_layer)
    c.add_port(
        name="E0", midpoint=[length, 0], width=width, orientation=0, layer=wg_layer
    )

    return c


if __name__ == "__main__":
    c = taper(width2=1, layers_cladding=[pp.LAYER.WGCLAD])
    # c = taper_strip_to_ridge()
    # print(c.get_optical_ports())
    # c = taper_strip_to_ridge_trenches()
    pp.show(c)
