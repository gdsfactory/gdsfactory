"""
Fab D

has a port naming convention with optical ports with o_ prefix and electrical ports with e_ prefix

"""

from typing import Callable

import pydantic.dataclasses as dataclasses

import gdsfactory as gf
from gdsfactory.add_pins import add_pin_square_inside
from gdsfactory.component import Component
from gdsfactory.cross_section import strip
from gdsfactory.port import select_ports
from gdsfactory.tech import LayerLevel, LayerStack, Library, Tech
from gdsfactory.types import Layer


@dataclasses.dataclass(frozen=True)
class LayerMap:
    WG: Layer = (10, 1)
    WG_CLAD: Layer = (10, 2)
    WGN: Layer = (34, 0)
    WGN_CLAD: Layer = (36, 0)
    PIN: Layer = (100, 0)


LAYER = LayerMap()
WIDTH_NITRIDE_OBAND = 0.9
WIDTH_NITRIDE_CBAND = 1.0

select_ports_optical = gf.partial(select_ports, layers_excluded=((100, 0),))


def get_layer_stack_fab_d(thickness_nm: float = 350.0) -> LayerStack:
    """Returns generic LayerStack"""
    return LayerStack(
        layers=[
            LayerLevel(
                name="core",
                gds_layer=34,
                gds_datatype=0,
                thickness_nm=350.0,
                zmin_nm=220.0 + 100.0,
            ),
            LayerLevel(name="clad", gds_layer=36, gds_datatype=0),
        ]
    )


def add_pins(
    component: Component,
    function: Callable = add_pin_square_inside,
    pin_length: float = 0.5,
    port_layer: Layer = LAYER.PIN,
    **kwargs,
) -> None:
    """Add Pin port markers.

    Args:
        component: to add ports
        function:
        pin_length:
        port_layer:
        function: kwargs

    """
    for p in component.ports.values():
        function(
            component=component,
            port=p,
            layer=port_layer,
            label_layer=port_layer,
            pin_length=pin_length,
            **kwargs,
        )
    component.auto_rename_ports_with_prefix()


# cross_sections

fabc_nitride_cband = gf.partial(
    strip, width=WIDTH_NITRIDE_CBAND, layer=LAYER.WGN, layers_cladding=(LAYER.WGN_CLAD,)
)
fabc_nitride_oband = gf.partial(
    strip, width=WIDTH_NITRIDE_OBAND, layer=LAYER.WGN, layers_cladding=(LAYER.WGN_CLAD,)
)
fabc_nitride_cband.__name__ = "fab_nitridec"
fabc_nitride_oband.__name__ = "fab_nitrideo"


# LEAF COMPONENTS have pins

mmi1x2_nitride_c = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_CBAND,
    cross_section=fabc_nitride_cband,
    decorator=add_pins,
)
mmi1x2_nitride_o = gf.partial(
    gf.components.mmi1x2,
    width=WIDTH_NITRIDE_OBAND,
    cross_section=fabc_nitride_oband,
    decorator=add_pins,
)
bend_euler_c = gf.partial(
    gf.components.bend_euler, cross_section=fabc_nitride_cband, decorator=add_pins
)
straight_c = gf.partial(
    gf.components.straight, cross_section=fabc_nitride_cband, decorator=add_pins
)
bend_euler_o = gf.partial(
    gf.components.bend_euler, cross_section=fabc_nitride_oband, decorator=add_pins
)
straight_o = gf.partial(
    gf.components.straight, cross_section=fabc_nitride_oband, decorator=add_pins
)

gc_nitride_c = gf.partial(
    gf.components.grating_coupler_elliptical_te,
    grating_line_width=0.6,
    wg_width=WIDTH_NITRIDE_CBAND,
    layer=LAYER.WGN,
    decorator=add_pins,
)

# HIERARCHICAL COMPONENTS made of leaf components

ring_c = gf.partial(gf.c.ring_single)


TECH_FABC = Tech(name="fab_d")

# for testing
LIBRARY = Library(name="fab_d")
LIBRARY.register(
    [mmi1x2_nitride_c, mmi1x2_nitride_o, bend_euler_c, straight_c, gc_nitride_c, ring_c]
)
# LIBRARY.register(
#     mmi1x2_nitride_c=mmi1x2_nitride_c,
#     mmi1x2_nitride_o=mmi1x2_nitride_o,
#     bend_euler_c=bend_euler_c,
#     straight_c=straight_c,
#     gc_nitride_c=gc_nitride_c,
#     ring_c=ring_c
# )


if __name__ == "__main__":

    c = ring_c()
    # cc = gf.routing.add_fiber_single(
    #     component=c,
    #     grating_coupler=gc_nitride_c,
    #     cross_section=fabc_nitride_cband,
    #     optical_routing_type=1,
    #     straight_factory=straight_c,
    #     bend_factory=bend_euler_c,
    #     select_ports=select_ports_optical,
    # )
    # cc.show()
