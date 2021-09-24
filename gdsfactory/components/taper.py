from typing import Optional, Tuple

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.port import Port
from gdsfactory.types import CrossSectionFactory, Layer


@cell
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: Optional[float] = None,
    port: Optional[Port] = None,
    with_cladding_box: bool = True,
    cross_section: CrossSectionFactory = strip,
    **kwargs
) -> Component:
    """Linear taper.

    Args:
        length:
        width1: width of the west port
        width2: width of the east port
        port: can taper from a port instead of defining width1
        with_cladding_box: to avoid DRC acute angle errors in cladding
        cross_section:
        kwargs: cross_section settings

    .. plot::
      :include-source:

      import gdsfactory as gf

      c = gf.components.taper(width1=0.5, width2=5, length=3)
      c.plot()

    """
    x = cross_section(**kwargs)

    layers_cladding = x.info["layers_cladding"]
    layer = x.info["layer"]

    if isinstance(port, gf.Port) and width1 is None:
        width1 = port.width
    if width2 is None:
        width2 = width1

    y1 = width1 / 2
    y2 = width2 / 2

    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]

    c = gf.Component()
    c.add_polygon((xpts, ypts), layer=layer)
    c.add_port(
        name="o1",
        midpoint=[0, 0],
        width=width1,
        orientation=180,
        layer=layer,
        cross_section=x,
    )
    c.add_port(
        name="o2",
        midpoint=[length, 0],
        width=width2,
        orientation=0,
        layer=layer,
        cross_section=x,
    )

    if with_cladding_box and x.info["layers_cladding"]:
        layers_cladding = x.info["layers_cladding"]
        cladding_offset = x.info["cladding_offset"]
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            top=cladding_offset,
        )
        for layer in layers_cladding or []:
            c.add_polygon(points, layer=layer)

    c.info["length"] = length
    c.info["width1"] = width1
    c.info["width2"] = width2
    return c


@gf.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.15,
    w_slab2: float = 6.0,
    layer_wg: Layer = gf.LAYER.WG,
    layer_slab: Layer = gf.LAYER.SLAB90,
    with_slab_port: bool = False,
    layers_cladding: Optional[Tuple[Layer, ...]] = None,
    cladding_offset: float = 3.0,
) -> Component:
    r"""Linear taper from strip to rib

    Args:
        length:
        width1:
        width2:
        w_slab1
        w_slab2
        layer_wg:
        layer_slab:
        with_slab_port: adds a wide slab port
        layers_cladding
        cladding_offset:

    .. code::

                      __________________________
                     /           |
             _______/____________|______________
                   /             |
       width1     |w_slab1       | w_slab2  width2
             ______\_____________|______________
                    \            |
                     \__________________________

    """

    taper_wg = taper(length=length, width1=width1, width2=width2, layer=layer_wg)
    taper_slab = taper(length=length, width1=w_slab1, width2=w_slab2, layer=layer_slab)

    c = gf.Component()
    for _t in [taper_wg, taper_slab]:
        taper_ref = _t.ref()
        c.add(taper_ref)
        c.absorb(taper_ref)

    c.info["length"] = length
    c.add_port(name="o1", port=taper_wg.ports["o1"])
    if with_slab_port:
        c.add_port(name="o2", port=taper_slab.ports["o2"])
    else:
        c.add_port(name="o2", port=taper_wg.ports["o2"])

    if layers_cladding:
        points = get_padding_points(
            component=c,
            default=0,
            bottom=cladding_offset,
            top=cladding_offset,
        )
        for layer in layers_cladding:
            c.add_polygon(points, layer=layer)

    return c


@gf.cell
def taper_strip_to_ridge_trenches(
    length=10.0,
    width=0.5,
    slab_offset=3.0,
    trench_width=2.0,
    trench_layer=gf.LAYER.SLAB90,
    layer_wg=gf.LAYER.WG,
    trench_offset_after_wg=0.1,
):

    c = gf.Component()
    y0 = width / 2 + trench_width - trench_offset_after_wg
    yL = width / 2 + trench_width - trench_offset_after_wg + slab_offset

    # straight
    x = [0, length, length, 0]
    yw = [y0, yL, -yL, -y0]
    c.add_polygon((x, yw), layer=layer_wg)

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

    c.add_port(name="o1", midpoint=[0, 0], width=width, orientation=180, layer=layer_wg)
    c.add_port(
        name="o2", midpoint=[length, 0], width=width, orientation=0, layer=layer_wg
    )

    return c


if __name__ == "__main__":
    # c = taper(width2=1)
    c = taper_strip_to_ridge(with_slab_port=True, layers_cladding=((111, 0),))
    # print(c.get_optical_ports())
    # c = taper_strip_to_ridge_trenches()
    c.show()
