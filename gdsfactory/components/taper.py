from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.add_padding import get_padding_points
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import CrossSectionSpec, LayerSpec


@cell
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: Port | None = None,
    with_bbox: bool = True,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "strip",
    port_order_name: tuple | None = ("o1", "o2"),
    port_order_types: tuple | None = ("optical", "optical"),
    **kwargs,
) -> Component:
    """Linear taper.

    Deprecated, use gf.components.taper_cross_section instead

    Args:
        length: taper length.
        width1: width of the west port.
        width2: width of the east port.
        port: can taper from a port instead of defining width1.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        with_two_ports: includes a second port.
            False for terminator and edge coupler fiber interface.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
         port_order_name(tuple): Ordered tuple of port names. First port is default taper port, second name only if with_two_ports flags used.
        port_order_types(tuple): Ordered tuple of port types. First port is default taper port, second name only if with_two_ports flags used.
        kwargs: cross_section settings.
    """
    x = gf.get_cross_section(cross_section, **kwargs)
    layer = x.layer

    if isinstance(port, gf.Port) and width1 is None:
        width1 = port.width
    if width2 is None:
        width2 = width1

    c = gf.Component()

    length = snap_to_grid(length)

    y1 = width1 / 2
    y2 = width2 / 2
    x1 = x.copy(width=width1)
    x2 = x.copy(width=width2)
    xpts = [0, length, length, 0]
    ypts = [y1, y2, -y2, -y1]
    c.add_polygon((xpts, ypts), layer=layer)

    for section in x.sections:
        layer = section.layer
        y1 = section.width / 2
        y2 = y1 + (width2 - width1)
        ypts = [y1, y2, -y2, -y1]
        c.add_polygon((xpts, ypts), layer=layer)

    if x.cladding_layers and x.cladding_offsets:
        for layer, offset in zip(x.cladding_layers, x.cladding_offsets):
            y1 = width1 / 2 + offset
            y2 = width2 / 2 + offset
            ypts = [y1, y2, -y2, -y1]
            c.add_polygon((xpts, ypts), layer=gf.get_layer(layer))

    c.add_port(
        name=port_order_name[0],
        center=(0, 0),
        width=width1,
        orientation=180,
        layer=x.layer,
        cross_section=x1,
        port_type=port_order_types[0],
    )
    if with_two_ports:
        c.add_port(
            name=port_order_name[1],
            center=(length, 0),
            width=width2,
            orientation=0,
            layer=x.layer,
            cross_section=x2,
            port_type=port_order_types[1],
        )

    if with_bbox and length:
        padding = []
        for offset in x.bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(x.bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    c.info["length"] = length
    c.info["width1"] = float(width1)
    c.info["width2"] = float(width2)

    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


@gf.cell
def taper_strip_to_ridge(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float = 0.5,
    w_slab1: float = 0.15,
    w_slab2: float = 6.0,
    layer_wg: LayerSpec = "WG",
    layer_slab: LayerSpec = "SLAB90",
    cross_section: CrossSectionSpec = "strip",
    bbox_layers: list[LayerSpec] | None = None,
    bbox_offsets: list[float] | None = None,
) -> Component:
    r"""Linear taper from strip to rib.

    Deprecated, use gf.components.taper_cross_section instead.

    Args:
        length: taper length (um).
        width1: in um.
        width2: in um.
        w_slab1: slab width in um.
        w_slab2: slab width in um.
        layer_wg: for input waveguide.
        layer_slab: for output waveguide with slab.
        cross_section: for input waveguide.

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
    taper_wg = taper(
        length=length,
        width1=width1,
        width2=width2,
        layer=layer_wg,
        cross_section=cross_section,
    )
    taper_slab = taper(
        length=length,
        width1=w_slab1,
        width2=w_slab2,
        layer=layer_slab,
    )

    c = gf.Component()
    for _t in [taper_wg, taper_slab]:
        taper_ref = _t.ref()
        c.add(taper_ref)
        c.absorb(taper_ref)

    c.info["length"] = length
    c.add_port(name="o1", port=taper_wg.ports["o1"])
    c.add_port(name="o2", port=taper_slab.ports["o2"])

    x = gf.get_cross_section(cross_section)
    if length:
        padding = []
        bbox_offsets = bbox_offsets or x.bbox_offsets
        bbox_layers = bbox_layers or x.bbox_layers

        for offset in bbox_offsets:
            points = get_padding_points(
                component=c,
                default=0,
                bottom=offset,
                top=offset,
            )
            padding.append(points)

        for layer, points in zip(bbox_layers, padding):
            c.add_polygon(points, layer=layer)

    if x.add_bbox:
        c = x.add_bbox(c)
    if x.add_pins:
        c = x.add_pins(c)
    return c


@gf.cell
def taper_strip_to_ridge_trenches(
    length: float = 10.0,
    width: float = 0.5,
    slab_offset: float = 3.0,
    trench_width: float = 2.0,
    trench_layer: LayerSpec = "DEEP_ETCH",
    layer_wg: LayerSpec = "WG",
    trench_offset: float = 0.1,
) -> gf.Component:
    """Defines taper using trenches to define the etch.

    Args:
        length: in um.
        width: in um.
        slab_offset: in um.
        trench_width: in um.
        trench_layer: trench layer.
        layer_wg: waveguide layer.
        trench_offset: after waveguide in um.
    """
    c = gf.Component()
    y0 = width / 2 + trench_width - trench_offset
    yL = width / 2 + trench_width - trench_offset + slab_offset

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

    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=layer_wg)
    c.add_port(
        name="o2", center=(length, 0), width=width, orientation=0, layer=layer_wg
    )
    return c


taper_strip_to_slab150 = partial(taper_strip_to_ridge, layer_slab="SLAB150")

# taper StripCband to NitrideCband
taper_sc_nc = partial(
    taper_strip_to_ridge,
    layer_wg="WG",
    layer_slab="WGN",
    length=20.0,
    width1=0.5,
    width2=0.15,
    w_slab1=0.15,
    w_slab2=1.0,
)


if __name__ == "__main__":
    # c = gf.grid(
    #     [
    #         taper_sc_nc(
    #             length=length, info=dict(doe="taper_length", taper_length=length)
    #         )
    #         for length in [1, 2, 3]
    #     ]
    # )
    # c.("extra/tapers.gds")
    xs = gf.get_cross_section("rib_conformal")

    c = taper(width2=1, cross_section="rib_conformal")
    # c = taper_strip_to_ridge()
    # print(c.get_optical_ports())
    # c = taper_strip_to_ridge_trenches()
    # c = taper()
    # c = gf.components.taper_strip_to_ridge(width1=1, width2=2)
    # c = gf.components.taper_strip_to_ridge(width1=1, width2=2)
    # c = gf.components.extend_ports(c)
    # c = taper_strip_to_ridge_trenches()
    # c = taper()
    # c = taper_sc_nc()
    c.show(show_ports=False)
