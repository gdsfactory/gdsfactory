"""FabC example."""

from __future__ import annotations

import pathlib
from functools import partial
from typing import Callable

from pydantic import BaseModel

import gdsfactory as gf
from gdsfactory.add_pins import add_pin_rectangle_inside
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports
from gdsfactory.technology import LayerLevel, LayerStack
from gdsfactory.typings import Layer


class LayerMap(BaseModel):
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (1, 10)


LAYER = LayerMap()
WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0

select_ports_optical = partial(select_ports, layers_excluded=((100, 0),))


def get_layer_stack_fab_c(thickness: float = 350.0) -> LayerStack:
    """Returns generic LayerStack."""
    return LayerStack(
        layers=dict(
            wg=LayerLevel(
                layer=LAYER.WG,
                zmin=0.0,
                thickness=0.22,
            ),
            wgn=LayerLevel(
                layer=LAYER.WGN,
                zmin=0.22 + 0.1,
                thickness=0.4,
            ),
        )
    )


def add_pins(
    component: Component,
    function: Callable = add_pin_rectangle_inside,
    pin_length: float = 0.5,
    port_layer: Layer = LAYER.PIN,
    **kwargs,
) -> Component:
    """Add Pin port markers.

    Args:
        component: to add ports.
        function: to add pins.
        pin_length: pin length in um.
        port_layer: for port.
        function: kwargs.

    """
    for p in component.ports.values():
        function(
            component=component,
            port=p,
            layer=port_layer,
            layer_label=port_layer,
            pin_length=pin_length,
            **kwargs,
        )
    return component


# cross_sections

xs_nc = partial(
    strip,
    width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    bbox_layers=[LAYER.WGN_CLAD],
    bbox_offsets=[3],
    add_pins=add_pins,
)
xs_no = partial(
    strip,
    width=WIDTH_NITRIDE_OBAND,
    layer=LAYER.WGN,
    bbox_layers=[LAYER.WGN_CLAD],
    bbox_offsets=[3],
    add_pins=add_pins,
)

cross_sections = dict(xs_nc=xs_nc, xs_no=xs_no)


# LEAF COMPONENTS have pins
bend_euler_nc = partial(gf.components.bend_euler, cross_section=xs_nc, with_bbox=True)
straight_nc = partial(gf.components.straight, cross_section=xs_nc, with_bbox=True)
bend_euler_o = partial(gf.components.bend_euler, cross_section=xs_no, with_bbox=True)
straight_o = partial(gf.components.straight, cross_section=xs_no, with_bbox=True)

mmi1x2_nc = partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    width_mmi=3,
    cross_section=xs_nc,
)
mmi1x2_no = partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    cross_section=xs_no,
)

gc_nc = partial(
    gf.components.grating_coupler_elliptical,
    grating_line_width=0.6,
    layer_slab=None,
    cross_section=xs_nc,
)

# HIERARCHICAL COMPONENTS made of leaf components

mzi_nc = partial(
    gf.components.mzi,
    cross_section=xs_nc,
    splitter=mmi1x2_nc,
    straight=straight_nc,
    bend=bend_euler_nc,
)
mzi_no = partial(
    gf.components.mzi,
    cross_section=xs_no,
    splitter=mmi1x2_no,
    straight=straight_o,
    bend=bend_euler_o,
)


# for testing
cells = dict(
    mmi1x2_nc=mmi1x2_nc,
    mmi1x2_no=mmi1x2_no,
    bend_euler_nc=bend_euler_nc,
    straight_nc=straight_nc,
    mzi_nc=mzi_nc,
    mzi_no=mzi_no,
    gc_nc=gc_nc,
)


LAYER_STACK = get_layer_stack_fab_c()
SPARAMETERS_PATH = pathlib.Path.home() / "fabc"

pdk = gf.Pdk(name="fab_c_demopdk", cells=cells, cross_sections=cross_sections)


if __name__ == "__main__":
    c1 = mmi1x2_nc()
    print(c1.name)
    # c1 = mmi1x2_nc()
    # d1 = c1.to_dict()

    # c2 = mmi1x2_nc(cache=False)
    # d2 = c2.to_dict()

    # from jsondiff import diff

    # d = diff(d1, d2)
    # c.show(show_ports=True)

    # c = mzi_nc()
    # print(c.name)
    # c.show()

    # mzi.show()
    # mzi_gc = gf.routing.add_fiber_single(
    #     component=mzi,
    #     grating_coupler=gc_nc,
    #     cross_section=xs_nc,
    #     optical_routing_type=1,
    #     straight=straight_nc,
    #     bend=bend_euler_nc,
    #     select_ports=select_ports_optical,
    # )
    # mzi_gc.show(show_ports=True)
