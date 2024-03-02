from __future__ import annotations

from functools import partial

import gdsfactory as gf
from gdsfactory.cell import cell
from gdsfactory.component import Component
from gdsfactory.port import Port
from gdsfactory.snap import snap_to_grid
from gdsfactory.typings import Callable, CrossSectionSpec, LayerSpec


@cell
def taper(
    length: float = 10.0,
    width1: float = 0.5,
    width2: float | None = None,
    port: Port | None = None,
    with_bbox: bool = True,
    with_two_ports: bool = True,
    cross_section: CrossSectionSpec = "xs_sc",
    port_order_name: tuple | None = ("o1", "o2"),
    port_order_types: tuple | None = ("optical", "optical"),
    add_pins: bool = True,
    post_process: Callable | None = None,
    **kwargs,
) -> Component:
    """Linear taper, which tapers only the main cross section section.

    Deprecated, use gf.components.taper_cross_section instead

    Args:
        length: taper length.
        width1: width of the west/left port.
        width2: width of the east/right port. Defaults to width1.
        port: can taper from a port instead of defining width1.
        with_bbox: box in bbox_layers and bbox_offsets to avoid DRC sharp edges.
        with_two_ports: includes a second port.
            False for terminator and edge coupler fiber interface.
        cross_section: specification (CrossSection, string, CrossSectionFactory dict).
        port_order_name(tuple): Ordered tuple of port names. First port is default \
                taper port, second name only if with_two_ports flags used.
        port_order_types(tuple): Ordered tuple of port types. First port is default \
                taper port, second name only if with_two_ports flags used.
        add_pins: add pins to the component.
        post_process: function to post process the component.
        kwargs: cross_section settings.
    """
    c = gf.Component()

    width1 = gf.snap.snap_to_grid2x(width1)
    x1 = gf.get_cross_section(cross_section, width=width1)
    if width2:
        width2 = gf.snap.snap_to_grid2x(width2)
        x2 = gf.get_cross_section(cross_section, width=width2)
    else:
        x2 = x1

    width1 = x1.width
    width2 = x2.width

    width_max = max([width1, width2])
    x = gf.get_cross_section(cross_section, width=width_max, **kwargs)
    layer = x.layer

    if isinstance(port, gf.Port) and width1 is None:
        width1 = port.width

    delta_width = width2 - width1

    length = float(snap_to_grid(length))
    y1 = width1 / 2
    y2 = width2 / 2

    if length:
        xpts = [0, length, length, 0]
        ypts = [y1, y2, -y2, -y1]
        c.add_polygon((xpts, ypts), layer=layer)

        xpts = [0, length, length, 0]
        for section in x.sections[1:]:
            layer = section.layer
            if not section.offset:
                y1 = section.width / 2
                y2 = section.width / 2 + delta_width / 2
                ypts = [y1, y2, -y2, -y1]
                c.add_polygon((xpts, ypts), layer=layer)
            else:
                y1 = section.width / 2
                y2 = section.width / 2
                y2 = section.width / 2 + delta_width / 2
                ypts = [y1, y2, -y2, -y1]
                ypts = [y - section.offset for y in ypts]
                c.add_polygon((xpts, ypts), layer=layer)

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
        x.add_bbox(c)
    if add_pins and x.add_pins:
        x.add_pins(c)
    if post_process:
        post_process(c)

    c.info["length"] = float(length)
    c.info["width1"] = float(width1)
    c.info["width2"] = float(width2)
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
    cross_section: CrossSectionSpec = "xs_sc",
    post_process: Callable | None = None,
    **kwargs,
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
        post_process: function to post process the component.
        kwargs: cross_section settings.

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

    width1 = gf.snap.snap_to_grid2x(width1)
    width2 = gf.snap.snap_to_grid2x(width2)
    w_slab1 = gf.snap.snap_to_grid2x(w_slab1)
    w_slab2 = gf.snap.snap_to_grid2x(w_slab2)

    xs = gf.get_cross_section(cross_section, **kwargs)
    xs_wg = gf.get_cross_section(
        cross_section, layer=layer_wg, add_pins_function_name=None
    )
    xs_slab = gf.get_cross_section(
        cross_section, layer=layer_slab, add_pins_function_name=None
    )

    taper_wg = taper(
        length=length,
        width1=width1,
        width2=width2,
        cross_section=xs_wg,
    )
    taper_slab = taper(
        length=length,
        width1=w_slab1,
        width2=w_slab2,
        cross_section=xs_slab,
    )

    c = gf.Component()
    for _t in [taper_wg, taper_slab]:
        taper_ref = _t.ref()
        c.add(taper_ref)
        c.absorb(taper_ref)

    c.info["length"] = float(length)
    c.add_port(name="o1", port=taper_wg.ports["o1"])
    c.add_port(name="o2", port=taper_slab.ports["o2"])
    # Add pins instead only on the final component
    if xs.add_pins:
        xs.add_pins(c)

    if length:
        xs.add_bbox(c)

    if post_process:
        post_process(c)

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
    post_process: Callable | None = None,
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
        post_process: function to post process the component.
    """
    c = gf.Component()
    y0 = width / 2 + trench_width - trench_offset
    yL = width / 2 + trench_width - trench_offset + slab_offset

    # straight
    x = [0, length, length, 0]
    yw = [y0, yL, -yL, -y0]
    if length:
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
    if post_process:
        post_process(c)
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
    c = taper(cross_section="xs_rc", width2=1, length=1)
    # xs_pin_m1 = partial(
    #     gf.cross_section.strip_auto_widen,
    #     width=0.5,
    #     width_wide=2,
    #     sections=(
    #         gf.Section(width=1, offset=2, layer=(24, 0), name="n+"),
    #         gf.Section(width=1, offset=3, layer=(41, 0), name="m1"),
    #     ),
    # )
    # # c = taper(width2=10, length=10)
    # # c = taper_strip_to_ridge_trenches()
    # # c = taper_strip_to_ridge()
    # # c = taper(width1=1.5, width2=1, cross_section="xs_rc")
    # # c = taper_sc_nc()
    # c = gf.Component("taper_with_offset")
    # route = gf.routing.get_route_from_waypoints(
    #     [(0, 0), (300, 0), (300, 300), (300, 600), (600, 600)],
    #     # cross_section="xs_sc_auto_widen",
    #     cross_section=xs_pin_m1,
    #     radius=30,
    # )
    # c.add(route.references)
    c.show(show_ports=False)
